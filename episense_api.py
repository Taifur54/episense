"""
EpiSense API Server
===================
Connects the AI detection engine to the dashboard.
Runs the model every 30 seconds and serves results via REST API.

Endpoints:
  GET /api/status     — system health
  GET /api/regions    — all region scores
  GET /api/region/:id — single region detail
  GET /api/alerts     — active alerts only
  POST /api/ingest    — receive new sensor data
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
import json
from datetime import datetime
from engine import DataGenerator, EpiSenseModel, REGIONS, SIGNAL_NAMES

app = Flask(__name__)
CORS(app)

# ── GLOBAL STATE ──
state = {
    "last_run": None,
    "results": [],
    "status": "initialising",
    "run_count": 0,
    "ingested_data": {},  # live sensor data overrides
}

gen = DataGenerator()
model = EpiSenseModel()

def initialise():
    """Train the model on historical data at startup."""
    print("[EpiSense API] Initialising...")
    history = gen.generate_history(days=90)
    model.train(history)
    state["status"] = "operational"
    print("[EpiSense API] Model ready. Starting scoring loop.\n")
    run_scoring()

def run_scoring():
    """Run the full scoring pipeline and update state."""
    current = gen.get_current_signals()

    # Merge any ingested live data
    for rid, data in state["ingested_data"].items():
        if rid in current:
            current[rid]["signals"].update(data)

    results = []
    for r in REGIONS:
        rid = r["id"]
        sigs = current[rid]["signals"]
        score = model.score_region(rid, sigs)
        level = model.get_alert_level(score["probability"])
        disease = current[rid]["disease_hint"]
        recs = model.generate_recommendations(
            r["name"], level, disease, score["signal_deviations"]
        )

        # Format signal deviations for API response
        formatted_sigs = {}
        for sig, dev in score["signal_deviations"].items():
            formatted_sigs[sig] = {
                "value": dev["current"],
                "baseline": dev["baseline_mean"],
                "change_pct": dev["deviation_pct"],
                "z_score": dev["z_score"],
                "flagged": dev["above_p95"],
            }

        results.append({
            **r,
            "probability": score["probability"],
            "alert_level": level,
            "disease_hint": disease,
            "signals": formatted_sigs,
            "recommendations": recs,
            "scored_at": datetime.utcnow().isoformat(),
        })

    results.sort(key=lambda x: x["probability"], reverse=True)
    state["results"] = results
    state["last_run"] = datetime.utcnow().isoformat()
    state["run_count"] += 1
    print(f"[EpiSense API] Scoring run #{state['run_count']} complete — "
          f"{sum(1 for r in results if r['alert_level']=='critical')} critical, "
          f"{sum(1 for r in results if r['alert_level']=='warning')} warning")

def scoring_loop():
    """Background thread: re-score every 30 seconds."""
    while True:
        time.sleep(30)
        run_scoring()

# ── ROUTES ──

@app.route("/api/status")
def status():
    return jsonify({
        "status": state["status"],
        "model": "EpiSense-v1.0",
        "regions_monitored": len(REGIONS),
        "signals_tracked": len(SIGNAL_NAMES),
        "last_run": state["last_run"],
        "run_count": state["run_count"],
        "critical_count": sum(1 for r in state["results"] if r["alert_level"] == "critical"),
        "warning_count": sum(1 for r in state["results"] if r["alert_level"] == "warning"),
        "timestamp": datetime.utcnow().isoformat(),
    })

@app.route("/api/regions")
def regions():
    level = request.args.get("level")  # filter by level
    data = state["results"]
    if level:
        data = [r for r in data if r["alert_level"] == level]
    return jsonify({
        "count": len(data),
        "regions": data,
        "generated_at": state["last_run"],
    })

@app.route("/api/region/<region_id>")
def region(region_id):
    match = next((r for r in state["results"] if r["id"] == region_id.upper()), None)
    if not match:
        return jsonify({"error": f"Region {region_id} not found"}), 404
    return jsonify(match)

@app.route("/api/alerts")
def alerts():
    active = [r for r in state["results"] if r["alert_level"] in ("critical", "warning", "watch")]
    return jsonify({
        "count": len(active),
        "alerts": active,
        "generated_at": state["last_run"],
    })

@app.route("/api/ingest", methods=["POST"])
def ingest():
    """
    Receive live sensor data from hardware nodes.
    Body: { "region_id": "IDN", "signals": { "pharmacy_index": 45.2, ... } }
    """
    data = request.json
    if not data or "region_id" not in data or "signals" not in data:
        return jsonify({"error": "Invalid payload. Required: region_id, signals"}), 400

    rid = data["region_id"].upper()
    state["ingested_data"][rid] = data["signals"]

    return jsonify({
        "accepted": True,
        "region_id": rid,
        "signals_received": list(data["signals"].keys()),
        "timestamp": datetime.utcnow().isoformat(),
    })

@app.route("/api/summary")
def summary():
    """High-level summary for dashboards and reports."""
    results = state["results"]
    return jsonify({
        "status": state["status"],
        "last_updated": state["last_run"],
        "global_threat_level": "critical" if any(r["alert_level"] == "critical" for r in results)
                               else "warning" if any(r["alert_level"] == "warning" for r in results)
                               else "watch" if any(r["alert_level"] == "watch" for r in results)
                               else "normal",
        "counts": {
            "critical": sum(1 for r in results if r["alert_level"] == "critical"),
            "warning": sum(1 for r in results if r["alert_level"] == "warning"),
            "watch": sum(1 for r in results if r["alert_level"] == "watch"),
            "normal": sum(1 for r in results if r["alert_level"] == "normal"),
            "total": len(results),
        },
        "top_threats": [
            {"name": r["name"], "probability": r["probability"],
             "level": r["alert_level"], "disease": r["disease_hint"]}
            for r in results[:5]
        ],
    })

if __name__ == "__main__":
    # Initialise model in background so server starts immediately
    threading.Thread(target=initialise, daemon=True).start()
    # Start scoring loop
    threading.Thread(target=scoring_loop, daemon=True).start()
    print("[EpiSense API] Server starting on http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)
