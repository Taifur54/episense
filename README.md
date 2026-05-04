# EpiSense — Global Pandemic Early Warning System

> Detect outbreaks 4–6 weeks before traditional surveillance. Built for governments that can't afford to wait.

**Live demo → [episense.github.io](https://episense.github.io)**

---

## What it does

EpiSense monitors anonymised signals from pharmacies, clinics, schools, and wastewater systems simultaneously across hundreds of cities. When multiple signals spike in the same region at the same time, our AI flags it as a potential outbreak — weeks before a doctor files a report.

During COVID-19, pharmacy sales spiked, schools closed, and wastewater viral loads elevated in Wuhan **6 weeks before WHO declared a global emergency**. EpiSense would have caught it.

## How it works

| Signal | Source | Lead time |
|---|---|---|
| Pharmacy sales | POS system APIs | 2–4 weeks |
| School absenteeism | Attendance management software | 1–3 weeks |
| Clinic visit rates | Hospital management systems | 1–2 weeks |
| Wastewater viral load | Municipal monitoring networks | 2–5 weeks |
| Social media symptoms | Public keyword clustering | 1–3 weeks |
| Travel anomalies | Flight and border data | Days |

## The AI engine

- **Model:** Isolation Forest (unsupervised anomaly detection) per region
- **Training:** 90-day clean baseline per region
- **Scoring:** 4-factor ensemble — anomaly score + weighted signal deviation + multi-signal convergence + max spike
- **Output:** Outbreak probability 0–100%, alert level, AI recommendations
- **Refresh:** Every 30 seconds

## Stack

- **Dashboard:** Vanilla HTML/CSS/JS — zero dependencies, works offline
- **AI Engine:** Python · scikit-learn · pandas · numpy
- **API:** Flask · REST · CORS-enabled
- **Data:** Simulated in MVP · Real pharmacy/clinic APIs in production

## Run locally

```bash
# Install dependencies
pip install numpy pandas scikit-learn flask flask-cors

# Start AI engine + API
python episense_api.py

# Open dashboard
open index.html
# Dashboard auto-connects to API at localhost:5050
```

## API endpoints

```
GET  /api/status       System health + model info
GET  /api/regions      All region scores
GET  /api/region/:id   Single region detail
GET  /api/alerts       Active alerts only
GET  /api/summary      Global threat level summary
POST /api/ingest       Receive live sensor data
```

## Funding

EpiSense is seeking seed funding to build the first real data partnerships and deploy pilots with health ministries. Relevant funders: YC, CEPI, Wellcome Trust, Gates Foundation Grand Challenges, USAID Global Health Security, World Bank Pandemic Fund.

---

Built by **Taifur Rahman** · [thingstoknow365@gmail.com](mailto:thingstoknow365@gmail.com)
