"""
EpiSense AI Detection Engine
============================
Ingests multi-signal health data streams, runs anomaly detection,
and outputs outbreak probability scores per region.

Signals tracked per region:
  - pharmacy_index     : pharmacy sales of fever/cough/antiviral meds (% above baseline)
  - clinic_load        : clinic visit rate vs 30-day rolling average
  - absenteeism        : school/workplace absenteeism rate
  - wastewater_score   : pathogen concentration in wastewater (0-100)
  - social_score       : symptom keyword frequency on social media (0-100)
  - travel_anomaly     : unusual inbound travel patterns (0-100)

Model: Isolation Forest (unsupervised anomaly detection) + weighted signal fusion
Output: outbreak_probability (0-100), alert_level, recommendations
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json
import random
import math

# ── REGION DEFINITIONS ──
REGIONS = [
    {"id": "IDN", "name": "Indonesia",     "region": "Southeast Asia",   "lat": -2.5,  "lng": 118.0},
    {"id": "NGA", "name": "Nigeria",       "region": "West Africa",      "lat": 9.0,   "lng": 8.0},
    {"id": "BRA", "name": "Brazil",        "region": "South America",    "lat": -10.0, "lng": -53.0},
    {"id": "IND", "name": "India",         "region": "South Asia",       "lat": 20.0,  "lng": 78.0},
    {"id": "EGY", "name": "Egypt",         "region": "North Africa",     "lat": 26.0,  "lng": 30.0},
    {"id": "MEX", "name": "Mexico",        "region": "North America",    "lat": 23.0,  "lng": -102.0},
    {"id": "PAK", "name": "Pakistan",      "region": "South Asia",       "lat": 30.0,  "lng": 69.0},
    {"id": "VNM", "name": "Vietnam",       "region": "Southeast Asia",   "lat": 16.0,  "lng": 108.0},
    {"id": "DEU", "name": "Germany",       "region": "Europe",           "lat": 51.0,  "lng": 10.0},
    {"id": "AUS", "name": "Australia",     "region": "Oceania",          "lat": -25.0, "lng": 134.0},
    {"id": "CAN", "name": "Canada",        "region": "North America",    "lat": 56.0,  "lng": -96.0},
    {"id": "ZAF", "name": "South Africa",  "region": "Southern Africa",  "lat": -29.0, "lng": 25.0},
    {"id": "BGD", "name": "Bangladesh",    "region": "South Asia",       "lat": 23.8,  "lng": 90.4},
    {"id": "PHL", "name": "Philippines",   "region": "Southeast Asia",   "lat": 12.9,  "lng": 121.8},
    {"id": "KEN", "name": "Kenya",         "region": "East Africa",      "lat": -0.5,  "lng": 37.9},
]

SIGNAL_NAMES = [
    "pharmacy_index",
    "clinic_load",
    "absenteeism",
    "wastewater_score",
    "social_score",
    "travel_anomaly",
]

# Signal weights — how much each contributes to outbreak probability
SIGNAL_WEIGHTS = {
    "pharmacy_index":   0.25,
    "clinic_load":      0.25,
    "absenteeism":      0.15,
    "wastewater_score": 0.20,
    "social_score":     0.10,
    "travel_anomaly":   0.05,
}

# ── DATA GENERATOR ──
# Simulates 90 days of historical data + live stream
# In production this would come from real pharmacy APIs,
# hospital systems, wastewater sensors, and social media

class DataGenerator:
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.baselines = {}
        self.outbreak_seeds = {}
        self._init_baselines()
        self._seed_outbreaks()

    def _init_baselines(self):
        """Each region has its own baseline signal levels."""
        for r in REGIONS:
            rid = r["id"]
            # Baseline varies by region (tropical = higher endemic baseline)
            tropical = r["lat"] < 25 and r["lat"] > -25
            base_mult = 1.3 if tropical else 1.0
            self.baselines[rid] = {
                "pharmacy_index":   random.uniform(8, 18) * base_mult,
                "clinic_load":      random.uniform(10, 20) * base_mult,
                "absenteeism":      random.uniform(5, 12) * base_mult,
                "wastewater_score": random.uniform(5, 15) * base_mult,
                "social_score":     random.uniform(5, 15) * base_mult,
                "travel_anomaly":   random.uniform(3, 10),
            }

    def _seed_outbreaks(self):
        """Outbreak start day (days before today = positive number)."""
        self.outbreak_seeds["IDN"] = {"started_days_ago": 14, "severity": 3.5, "disease": "Respiratory (H5N2-like)"}
        self.outbreak_seeds["NGA"] = {"started_days_ago": 7,  "severity": 3.0, "disease": "Haemorrhagic fever signal"}
        self.outbreak_seeds["BRA"] = {"started_days_ago": 21, "severity": 2.2, "disease": "Arbovirus (Dengue/Zika)"}
        self.outbreak_seeds["IND"] = {"started_days_ago": 10, "severity": 2.0, "disease": "Respiratory cluster"}
        self.outbreak_seeds["EGY"] = {"started_days_ago": 5,  "severity": 1.8, "disease": "Gastrointestinal signal"}

    def _outbreak_multiplier(self, region_id, days_before_today):
        """
        How much signals are amplified on a given historical day.
        days_before_today=0 means today, =90 means 90 days ago.
        """
        if region_id not in self.outbreak_seeds:
            return 1.0
        seed = self.outbreak_seeds[region_id]
        # How many days ago was that historical point relative to outbreak start?
        # outbreak started seed["started_days_ago"] days before today
        # historical point is days_before_today days before today
        # so elapsed days into outbreak = started_days_ago - days_before_today
        elapsed = seed["started_days_ago"] - days_before_today
        if elapsed <= 0:
            return 1.0  # Before outbreak started
        k = 0.25
        L = seed["severity"]
        mult = L / (1 + math.exp(-k * (elapsed - 8)))
        return 1.0 + mult

    def generate_history(self, days=90):
        """Generate 90 days of historical signal data per region."""
        records = []
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        for r in REGIONS:
            rid = r["id"]
            baseline = self.baselines[rid]

            for d in range(days, 0, -1):
                date = today - timedelta(days=d)
                mult = self._outbreak_multiplier(rid, d)

                row = {"region_id": rid, "date": date.isoformat()}
                for sig in SIGNAL_NAMES:
                    base = baseline[sig]
                    noise = np.random.normal(0, base * 0.08)
                    seasonal = base * 0.1 * math.sin(2 * math.pi * d / 365)
                    val = base * mult + noise + seasonal
                    row[sig] = round(max(0, min(100, val)), 2)

                records.append(row)

        return pd.DataFrame(records)

    def get_current_signals(self):
        """Get current (live) signal readings for all regions."""
        result = {}
        today = datetime.utcnow()

        for r in REGIONS:
            rid = r["id"]
            baseline = self.baselines[rid]
            mult = self._outbreak_multiplier(rid, 0)
            signals = {}
            for sig in SIGNAL_NAMES:
                base = baseline[sig]
                noise = np.random.normal(0, base * 0.05)
                val = base * mult + noise
                signals[sig] = round(max(0, min(100, val)), 2)

            result[rid] = {
                "signals": signals,
                "timestamp": today.isoformat(),
                "disease_hint": self.outbreak_seeds.get(rid, {}).get("disease", "No significant signals"),
            }

        return result


# ── AI MODEL ──

class EpiSenseModel:
    def __init__(self):
        self.models = {}      # One Isolation Forest per region
        self.scalers = {}     # One scaler per region
        self.histories = {}   # Historical baselines per region
        self.trained = False

    def train(self, historical_df):
        """
        Train on the CLEAN baseline period only (first 60 days).
        Isolation Forest learns what normal looks like —
        then current outbreak signals score as anomalies.
        """
        print("Training EpiSense anomaly detection model...")
        print(f"Training on {len(historical_df)} historical data points across {historical_df['region_id'].nunique()} regions\n")

        for rid in historical_df["region_id"].unique():
            region_data = historical_df[historical_df["region_id"] == rid]
            # Use only the first 60 days (clean baseline, before outbreaks)
            baseline_data = region_data.head(60)[SIGNAL_NAMES]

            scaler = StandardScaler()
            scaled = scaler.fit_transform(baseline_data)

            # Very low contamination — we trained on clean data
            model = IsolationForest(
                contamination=0.02,
                n_estimators=200,
                random_state=42,
            )
            model.fit(scaled)

            self.scalers[rid] = scaler
            self.models[rid] = model

            self.histories[rid] = {
                "mean": baseline_data.mean().to_dict(),
                "std":  baseline_data.std().to_dict(),
                "p90":  baseline_data.quantile(0.90).to_dict(),
                "p95":  baseline_data.quantile(0.95).to_dict(),
            }

        self.trained = True
        print(f"Model trained on {len(self.models)} regions (60-day clean baseline).")

    def score_region(self, region_id, current_signals):
        """
        Score current signals against historical baseline.
        Returns outbreak probability 0-100.
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() first.")

        if region_id not in self.models:
            return {"probability": 0, "anomaly_score": 0, "signal_deviations": {}}

        signals_array = np.array([[current_signals[s] for s in SIGNAL_NAMES]])
        scaled = self.scalers[region_id].transform(signals_array)

        # Isolation Forest score: negative = anomalous
        raw_score = self.models[region_id].score_samples(scaled)[0]

        # Convert to 0-100 probability
        # score_samples returns values roughly in [-0.5, 0.5]
        # More negative = more anomalous = higher outbreak probability
        anomaly_score = max(0, min(1, (-raw_score - 0.1) / 0.4))

        # Signal deviations from historical mean
        baseline = self.histories[region_id]
        deviations = {}
        for sig in SIGNAL_NAMES:
            mean = baseline["mean"][sig]
            std = baseline["std"].get(sig, 1)
            p95 = baseline["p95"][sig]
            current = current_signals[sig]
            deviation_pct = ((current - mean) / mean * 100) if mean > 0 else 0
            z_score = (current - mean) / std if std > 0 else 0
            deviations[sig] = {
                "current": round(current, 2),
                "baseline_mean": round(mean, 2),
                "deviation_pct": round(deviation_pct, 1),
                "z_score": round(z_score, 2),
                "above_p95": current > p95,
            }

        # Weighted signal fusion — how much each signal deviates
        weighted_score = sum(
            min(1, max(0, deviations[sig]["z_score"] / 3)) * SIGNAL_WEIGHTS[sig]
            for sig in SIGNAL_NAMES
        )

        # Count multi-signal convergence (more signals spiking = higher confidence)
        convergence_bonus = sum(
            1 for sig in SIGNAL_NAMES if deviations[sig]["above_p95"]
        ) / len(SIGNAL_NAMES)

        # Max z-score (single strong signal)
        max_z = max(deviations[sig]["z_score"] for sig in SIGNAL_NAMES)
        max_z_score = min(1, max(0, max_z / 5))

        # Ensemble: isolation forest + weighted deviation + convergence + max spike
        final_prob = (
            anomaly_score     * 0.30 +
            weighted_score    * 0.35 +
            convergence_bonus * 0.20 +
            max_z_score       * 0.15
        ) * 100

        # Non-linear scaling so outbreak regions hit 70-90%, normal stays 5-25%
        final_prob = final_prob ** 1.4 / (100 ** 0.4)
        final_prob = round(max(2, min(99, final_prob)), 1)

        return {
            "probability": final_prob,
            "anomaly_score": round(anomaly_score, 4),
            "weighted_score": round(weighted_score, 4),
            "signal_deviations": deviations,
        }

    def get_alert_level(self, probability):
        if probability >= 70: return "critical"
        if probability >= 40: return "warning"
        if probability >= 20: return "watch"
        return "normal"

    def generate_recommendations(self, region_name, alert_level, disease_hint, deviations):
        recs = []
        if alert_level == "critical":
            recs += [
                f"Deploy field investigation team to {region_name} immediately",
                "Notify WHO regional office within 6 hours",
                "Activate hospital surge capacity protocols",
                "Issue public health advisory for affected districts",
            ]
        elif alert_level == "warning":
            recs += [
                f"Alert {region_name} national health authority",
                "Increase surveillance frequency to every 6 hours",
                "Prepare rapid response team for potential deployment",
            ]
        elif alert_level == "watch":
            recs += [
                "Continue enhanced monitoring",
                f"Brief {region_name} public health officials",
                "No immediate action required — review in 48h",
            ]
        else:
            recs.append("No action required — all signals within normal range")

        # Signal-specific recommendations
        if deviations.get("wastewater_score", {}).get("above_p95"):
            recs.append("Wastewater signal elevated — check water treatment facilities")
        if deviations.get("pharmacy_index", {}).get("z_score", 0) > 3:
            recs.append("Pharmacy spike critical — map geographic distribution of sales")
        if deviations.get("absenteeism", {}).get("z_score", 0) > 2.5:
            recs.append("High absenteeism — consider school/workplace closure assessment")

        return recs[:5]  # Max 5 recommendations


# ── FULL PIPELINE RUN ──

def run_pipeline():
    print("=" * 60)
    print("  EpiSense AI Detection Engine v1.0")
    print("  Global Pandemic Early Warning System")
    print("=" * 60)
    print()

    # 1. Generate historical data
    print("Step 1: Generating historical signal data (90 days)...")
    gen = DataGenerator()
    history = gen.generate_history(days=90)
    print(f"  Generated {len(history):,} data points")
    print(f"  Signals: {', '.join(SIGNAL_NAMES)}")
    print()

    # 2. Train model
    print("Step 2: Training anomaly detection model...")
    model = EpiSenseModel()
    model.train(history)
    print()

    # 3. Get current live signals
    print("Step 3: Reading current signal feeds...")
    current = gen.get_current_signals()
    print(f"  Received signals from {len(current)} regions")
    print()

    # 4. Score all regions
    print("Step 4: Running outbreak probability scoring...")
    print()

    results = []
    for r in REGIONS:
        rid = r["id"]
        sigs = current[rid]["signals"]
        score = model.score_region(rid, sigs)
        level = model.get_alert_level(score["probability"])
        disease = current[rid]["disease_hint"]
        recs = model.generate_recommendations(r["name"], level, disease, score["signal_deviations"])

        result = {
            **r,
            "probability": score["probability"],
            "alert_level": level,
            "disease_hint": disease,
            "signals": sigs,
            "signal_deviations": score["signal_deviations"],
            "anomaly_score": score["anomaly_score"],
            "recommendations": recs,
            "timestamp": current[rid]["timestamp"],
        }
        results.append(result)

    # Sort by probability
    results.sort(key=lambda x: x["probability"], reverse=True)

    # 5. Print results
    print("━" * 60)
    print(f"  {'REGION':<20} {'PROB':>6}  {'LEVEL':<10} {'DISEASE'}")
    print("━" * 60)

    level_icons = {"critical": "🔴", "warning": "🟡", "watch": "🔵", "normal": "🟢"}
    for r in results:
        icon = level_icons[r["alert_level"]]
        name = r["name"][:18]
        prob = f"{r['probability']}%"
        level = r["alert_level"].upper()
        disease = r["disease_hint"][:35]
        print(f"  {name:<20} {prob:>6}  {level:<10} {disease}")

    print("━" * 60)
    print()

    # 6. Detailed view of top alerts
    critical = [r for r in results if r["alert_level"] == "critical"]
    warnings = [r for r in results if r["alert_level"] == "warning"]

    print(f"SUMMARY: {len(critical)} CRITICAL · {len(warnings)} WARNING · {len(results) - len(critical) - len(warnings)} other\n")

    if critical:
        print("CRITICAL ALERTS — IMMEDIATE ACTION REQUIRED")
        print("─" * 60)
        for r in critical:
            print(f"\n  {r['name'].upper()} ({r['region']})")
            print(f"  Outbreak probability: {r['probability']}%")
            print(f"  Disease signal: {r['disease_hint']}")
            print(f"  Key signals:")
            for sig, dev in list(r["signal_deviations"].items())[:4]:
                arrow = "↑" if dev["deviation_pct"] > 0 else "↓"
                flag = " *** CRITICAL ***" if dev["above_p95"] else ""
                print(f"    {sig:<20} {dev['current']:>6.1f}  ({arrow}{abs(dev['deviation_pct']):.0f}% vs baseline){flag}")
            print(f"  Recommendations:")
            for rec in r["recommendations"][:3]:
                print(f"    → {rec}")

    # 7. Save JSON output for dashboard
    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "model_version": "EpiSense-v1.0",
        "total_regions": len(results),
        "critical_count": len(critical),
        "warning_count": len(warnings),
        "regions": results,
    }

    with open("/home/claude/episense/output.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Output saved to output.json")
    print(f"  Ready to connect to dashboard API\n")

    return output


if __name__ == "__main__":
    output = run_pipeline()
