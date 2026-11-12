# AstroCare: AI Health Sentinel for Astronauts, from Launch to Recovery

**Team AstroZen · NASA Space Apps Challenge 2026**
Challenge: *Create Health Monitoring Software for Astronauts on Space Missions*

> **"Nothing is diagnosed. Everything is evidenced."**

AstroCare is an evidence-first crew health console. It brings space weather, radiation dose, blood radiation response, immune and metabolic lab panels, microbiome, telomere evidence, daily vital signs and self-reports into one place. It compares every astronaut with **their own normal**, not a population average.

Only two parts are true machine-learning models, because only they have enough suitable NASA data to train and validate honestly. **M1** forecasts solar particle storms and **M3** reads radiation dose from blood gene expression. Everything else measures change against each person's own pre-flight baseline with fixed, transparent rules, estimates dose from physics, or shows published context. Every value on screen carries its source, its rule and a safe, non-diagnostic next step.

| | |
|---|---|
| **9** health modules, launch to recovery | **2** machine-learning models (M1, M3) |
| **6** rule engines against each person's own baseline | **0** diagnoses: every flag shows its evidence |

Runs **fully offline** from one command (`python run.py`). No internet is needed at run time.

---

## Table of contents

- [Why AstroCare](#why-astrocare)
- [Design principles](#design-principles)
- [Features](#features)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Data](#data)
- [Methods and results](#methods-and-results)
- [Rules and recommendations](#rules-and-recommendations)
- [Provenance: seven status types](#provenance-seven-status-types)
- [Pages of the console](#pages-of-the-console)
- [API](#api)
- [Testing](#testing)
- [Limitations](#limitations)
- [Future work](#future-work)
- [Team](#team)
- [References](#references)

---

## Why AstroCare

Space is hard on the human body in ways the body does not announce. Radiation damages DNA without any sensation, microgravity weakens bone and changes the heart, and isolation affects sleep, mood and reaction time. On the way to Mars a message to a doctor on Earth can take up to **twenty minutes each way**, so crews will need to check their own health and act sensibly before help can answer.

Spaceflight health data are also scattered across space-weather systems, dosimetry archives, research repositories and wearables. Much health software either predicts without enough data or gives a single score that hides the evidence. AstroCare does the opposite: it predicts only where NASA data support a prediction, compares each astronaut with their own normal everywhere else, and always shows the rule and the source behind what it says.

**Users**

- **Primary:** astronauts and crew (on-board self-monitoring)
- **Secondary:** flight surgeons and mission medical operations (review of flags and evidence)
- **Tertiary:** researchers (provenance-labelled data, methods and code)

## Design principles

1. Compare each astronaut with their own normal, not a population threshold.
2. Warn early only where NASA data support a prediction; otherwise measure, show context, or ask the astronaut.
3. Never diagnose, never name a medication, never give one overall health score.
4. Show the evidence for every flag: what changed, when, which rule fired, and which study supports it.
5. Refuse to answer outside a model's validated window.
6. Run without Earth: no internet at run time.

Output language is limited to phrases such as *"Changed from baseline"*, *"Monitoring recommended"* and *"Back within own variation"*.

## Features

| # | Feature | Module | Method |
|---|---|---|---|
| 1 | Solar particle storm early warning (48 h) | M1 | Machine learning (calibrated logistic regression) |
| 2 | Crew radiation dose by location (habitat, shelter, EVA) | M2 | Physics estimate |
| 3 | Blood radiation biodosimeter: dose at 20–28 h; dose class at 1, 2, 6 h | M3 | Machine learning (elastic-net) |
| 4 | Microbiome drift from own pre-flight samples | M4 | Rule vs own baseline (M4-DRIFT-v1) |
| 5 | Brain-risk evidence card | M5 | Published context (animal studies) |
| 6 | Recovery after landing (R+1 to R+194) | M6 | Rule vs own baseline (M6-TRAJ-v2) |
| 7 | Immune and inflammatory monitoring (serum panels, blood counts) | M7 | Rule vs own baseline (M7-DEV-v1) |
| 8 | Telomere evidence card | M8 | Published context |
| 9 | Crew Health Console: daily vitals, live heart rate, flag feed | M9 | Rules vs own 30-day baseline (M9-FLAG-v1, M9-MULTI-v1, M9-Q01) |
| 10 | Mission context: phase, gravity, isolation, cabin CO2 | Context | Context only |
| 11 | Bone panel: change from own pre-flight DXA scan | Bone | Measured + derived; published context |
| 12 | Daily self-check (sleep, stress, mood, fatigue) and reaction-time test | Behaviour | Self-report + in-app measurement, rule R50 |
| 13 | Explanation card for every flag | All | What changed · when · rule · evidence |
| 14 | Action / Recommendation Center | All | Rules R02–R99 in priority order |
| 15 | Provenance record and evidence registry for every value | All | 7 status types; 14 evidence entries |

### The evidence gate: machine learning vs rules

A domain gets a machine-learning model only when suitable human (or human-cell) data, a real target and enough independent subjects exist. That is true for solar storms (decades of events) and blood biodosimetry (ex vivo irradiated human blood from 5–11 donors per model). The Inspiration4 crew data cover four people, which is too few to train and validate a model honestly, so those domains use deterministic rules against each person's own pre-flight baseline, an approach that stays valid even for a single person.

| Domain | Human data | Tier | Module |
|---|---|---|---|
| Solar storm forecasting | DONKI + GOES event history 2010–2026 | Tier 1 · ML | M1 |
| Crew dose | MSL/RAD background rate | Estimate | M2 |
| Blood radiation response | OSD-1012 (24 h), OSD-410 (1/2/6 h); OSD-71 external test | Tier 1 · ML | M3 |
| Microbiome | OSD-572 swabs, 4 crew, Tier A sites | Tier 2 · trend | M4 |
| Brain | Mouse studies only | Tier 3 · context | M5 |
| Recovery | OSD-575 + OSD-569 panels to R+194 | Tier 2 · trend | M6 |
| Immunology | OSD-575 Eve + NULISA, OSD-569 CBC, 4 crew | Tier 2 · trend | M7 |
| Telomere | Published group results only | Tier 3 · context | M8 |
| Vital signs | Mission telemetry; rule from Iglewicz & Hoaglin 1993 | Tier 4 · rules | M9 |
| Bone | Published recovery parameters; scans if entered | Tier 3 · context | Bone panel |
| Behaviour | Self-report + in-app reaction test | Tier 4 · self-report | Behaviour |

---

## Quick start

**Requirements:** Python 3.9 or newer. Nothing else is required.

```bash
# 1. Open the astrocare folder in a terminal (or in VS Code: File → Open Folder)
cd astrocare

# 2. Start the console
python run.py
```

Your browser opens **http://127.0.0.1:8000** automatically. Stop the server with **Ctrl + C**.
On Windows, use `py` if `python` is not found.

> Do not open `frontend/index.html` directly by double-clicking it. The pages load their data from the backend, so `python run.py` must be running.

### Optional: FastAPI server with interactive API docs

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Interactive API documentation is then available at **http://127.0.0.1:8000/docs**. If FastAPI is not installed, `run.py` falls back to a built-in standard-library server with the identical API.

### Command-line options

| Option | Effect |
|---|---|
| `--port 8080` | Use another port (default 8000) |
| `--host 0.0.0.0` | Bind to another host (default 127.0.0.1) |
| `--stdlib` | Force the built-in server even if FastAPI is installed |
| `--no-browser` | Do not open the browser automatically |

Environment variables `ASTROCARE_HOST`, `ASTROCARE_PORT` and `ASTROCARE_RUNTIME_DIR` override the defaults in `backend/app/config.py`.

### VS Code

The **Run and Debug** panel includes three ready-made launch configurations: run the server, run the tests, and rebuild the simulated vitals.

---

## Architecture

AstroCare has four layers: a **data layer** of NASA and published datasets stored with the software; an **analysis layer** of two ML models, one physics estimate and six rule engines; an **API layer** that turns every result into a provenance-labelled indicator; and a **12-page web console** for the crew and the flight surgeon.

```
data files → module (model / rule / context) → indicator record with status_type
           → explanation → action engine → page
```

### Technology stack

| Layer | Technology | What it does |
|---|---|---|
| Analysis notebooks | Python, Google Colab (pandas, scikit-learn, limma/oligo where needed) | Data cleaning, model training and validation for M1, M3, M4, M9 |
| Backend | Python 3.9+; FastAPI + uvicorn (optional); built-in standard-library server as fallback | 33 API endpoints; models and rules run in pure Python (no scikit-learn needed at run time) |
| Rule set | Versioned JSON (`rules_v1.json`), evidence registry (14 entries) | All thresholds, rule texts and evidence links in one place |
| Frontend | HTML, CSS, JavaScript; canvas charts; Three.js 3D hero scene | 12 pages |
| Testing | Python `unittest` (26 tests) + headless-browser checks | Reproduction of notebook results, safety wording, provenance, every page and endpoint |

### Project structure

```
astrocare/
├── run.py                       start here (FastAPI if installed, else built-in server)
├── requirements.txt             optional: fastapi, uvicorn, numpy
├── backend/
│   ├── app/
│   │   ├── config.py            app name, paths, host and port
│   │   ├── routes.py            one route table used by both servers
│   │   ├── server_fastapi.py    FastAPI app (/docs)
│   │   ├── server_stdlib.py     zero-install server
│   │   ├── provenance.py        indicator record + the 7 status types
│   │   ├── indicators.py        every displayed value for a subject
│   │   ├── store.py             cached file loading, runtime storage
│   │   ├── modules/
│   │   │   ├── m1_storm.py      calibrated logistic regression (coefficients exported from joblib)
│   │   │   ├── m2_dose.py       physics estimate
│   │   │   ├── m3_blood.py      pure-Python port of the M3 predictor + window rule
│   │   │   ├── m9_vitals.py     M9-FLAG-v1, M9-MULTI-v1, M9-Q01
│   │   │   ├── i4_lab.py        M7-DEV-v1 and M6-TRAJ-v2 rules on the lab panels
│   │   │   ├── inspiration4.py  M4, M5, M6, M7, M8 pages
│   │   │   ├── body_mind.py     self-report, reaction test, bone
│   │   │   └── context.py       mission timeline, cabin CO2
│   │   └── rules/action_engine.py   rule-based recommendations
│   ├── data/
│   │   ├── reference/           rules_v1.json, evidence_registry.json, mission.json, m8_context.json
│   │   ├── m1/ m3/ m4/ m9/      model exports, validation files, cards
│   │   ├── i4/                  Inspiration4 lab panels (OSD-575, OSD-569) + SOURCES.md + SHA256SUMS
│   │   ├── cards/               optional notebook cards
│   │   └── runtime/             self-reports, scans, reaction tests (created at run time)
│   ├── tools/simulate_vitals.py rebuilds the M9 mission telemetry
│   └── tests/test_system.py     26 tests
└── frontend/
    ├── index.html               12-page single-page app
    ├── css/app.css
    ├── js/core.js charts.js hero3d.js app.js
    ├── js/views/*.js            one file per page
    ├── js/vendor/three.min.js   Three.js
    └── assets/img/              M4 and M9 notebook figures
```

### Two crews, never mixed

- **Mission crew (SIM-A, SIM-B, SIM-C, SIM-D):** a four-person deep-space mission scenario from mission day −45 to +45. Drives the Console (M9), Radiation (M1/M2), Blood (M3) and Body & Mind pages.
- **Inspiration4 crew (C001–C004):** the de-identified NASA OSDR subjects. Drive Microbiome (M4), Immune (M7), Telomere (M8) and Recovery (M6).

The crew selector switches automatically with the page, so measurements from one group are never shown under the other's name.

---

## Data

Raw files were never edited. Each module has its own cleaned tables, and every displayed value can be traced back to a file and a processing step.

| Module | Dataset | Subjects / size | Used as |
|---|---|---|---|
| M1 | NASA DONKI flare + SEP catalogue; NOAA GOES ≥10 MeV proton flux | Flares 2010–2026; train 1,167 flares (39 storms), test 1,720 forecast points (21 storms) | Training and test data |
| M2 | MSL/RAD (Curiosity cruise to Mars), Zeitlin et al. 2013 | GCR dose-equivalent ≈ 1.8 mSv/day | Physical parameter |
| M3 | OSDR OSD-1012 (GSE90909) | 12 donors (11 usable); X-ray 0–4 Gy, neutrons 0.25–1 Gy; 24 h | Training (24 h) + neutron test |
| M3 | OSDR OSD-410 | 10 donors; 0 / 0.15 / 0.30 / 1.5 Gy γ; 1, 2, 6 h | Training (early models) |
| M3 | OSDR OSD-71 | 4 donors; ~1 h; different lab and platform | External test only |
| M4 | OSDR OSD-572 (Inspiration4 swabs) | 4 crew, 97 samples, 8 timepoints | Rule input |
| M6, M7 | OSDR OSD-575 serum: Eve 71-plex, Alamar NULISA 203-target, Eve cardiovascular, CMP | 4 crew × 7 timepoints (L-92 … R+194) | Rule input |
| M6, M7 | OSDR OSD-569 whole blood: CBC | 4 crew × 7 timepoints | Rule input |
| M8 | Garcia-Medina et al. 2024 (PMC11022651) | 4 crew; dried blood spots | Context |
| M5 | OSDR OSD-611, OSD-374, OSD-479 | Mice | Context |
| M9 | Mission crew telemetry SIM-A … SIM-D | 4 crew × 4 vitals × 90 mission days | Rule input |
| Bone | NASA skeletal recovery study (NTRS 20070001978) | Long-duration crew | Context |
| Behaviour | In-app entries | Per crew member | Self-report / measured |

**Data notes**

- **M3:** the three blood studies were never pooled; each model is trained on one study. Every exposed sample is compared with the same donor's sham (0 Gy) sample. Earlier exports had been silently cut at the Excel row limit (1,048,575 rows), so the cleaned file is never opened in spreadsheet software.
- **M4:** sites were tiered; only Tier A ("reliable") sites are displayed. Three environmental taxa appear in about 70% of in-flight swabs but only 0–2% of pre-flight swabs, so they are shown as a likely-contamination QC note.
- **Inspiration4 lab panels:** OSDR was not reachable from the build environment, so the files were taken from two public GitHub mirrors and cross-checked (see `backend/data/i4/SOURCES.md`). SHA-256 checksums of every file are verified by a test. Race-coded eGFR fields were removed.
- **M9:** the M9 notebook shipped only its 14 flag cards, not the daily series. `backend/tools/simulate_vitals.py` rebuilds the series from the medians and MADs in those cards so that the rule reproduces **exactly the notebook's 14 flags and no others**. To use a real series, export it as `backend/data/m9/m9_vitals_daily.csv` with the columns `crew_id, mission_day, date, measure, value` (measure = `hr` / `hrv` / `spo2` / `rr`).

---

## Methods and results

### M1: solar particle storm forecaster (machine learning)

- **Model:** StandardScaler → logistic regression (L2, balanced classes), then Platt calibration.
- **Feature:** `flare_log10_peak_flux`. Flare size alone had the best precision–recall area in cross-validation on the training period, beating 6- and 7-feature models.
- **Alert:** calibrated probability ≥ 0.05, roughly a flare of class M5.0 or larger.
- **Split by time:** train 2010-06-12 to 2023-12-24; test 2024-01-01 to 2026-10-19.

| Test metric (2024–2026) | Value |
|---|---|
| Storms warned / missed | 14 / 7 |
| Recall (95% CI) | 66.7% (44.4–87%) |
| ROC-AUC | 0.811 |
| PR-AUC (95% CI) | 0.084 (0.041–0.171) |
| False alerts | 247 (≈ 7.3 per month) |
| Brier score | 0.013 |
| May 2024 storm (X3.9 flare, 10 May) | warned, p = 0.349 |

### M2: crew dose estimate (physics)

Daily dose-equivalent = 1.8 mSv/day galactic cosmic-ray background + k × (≥10 MeV proton flux above background) × shielding factor, with k = 0.0006 mSv/day per pfu and shielding factors habitat 1.0, storm shelter 0.35, EVA 3.5. Cumulative dose is shown as a share of NASA's 600 mSv career limit (NASA-STD-3001). Over 45 in-flight days: **83.22 mSv for SIM-A** and **85.54 mSv for SIM-D** (on EVA at storm onset).

### M3: blood radiation biodosimeter (machine learning)

Four separate models, one per time window, all validated leave-one-donor-out with nested grid search. The model only ever sees *exposed − own baseline* per gene.

| Model | Training data | Task | Result |
|---|---|---|---|
| M3-24h-v2 | OSD-1012 X-ray, 11 donors | Elastic-net regression, 11 genes | MAE 0.379 Gy, 81.6% within 0.5 Gy |
| M3-early-1h-v2 | OSD-410 γ, 10 donors | 4 dose classes, 50 genes | Balanced accuracy 0.60 (chance 0.33) |
| M3-early-2h-v2 | OSD-410 γ, 10 donors | 4 dose classes, 50 genes | Balanced accuracy 0.60 (chance 0.33) |
| M3-early-6h-v2 | OSD-410 γ, 5 donors | 4 dose classes, 20 genes | Balanced accuracy 0.80 (chance 0.33) |

**Safeguards:** valid range 0.1–2 Gy at 20–28 h (4 Gy is under-read); neutrons are over-read about 2.6×, so readings are labelled **X-ray-equivalent**; the 1 h model failed the external OSD-71 test and is marked exploratory; rule R02 refuses any estimate outside the validated windows or without a baseline sample. The stable 24 h genes include known p53-pathway radiation-response genes (FDXR, DDB2, GADD45A, PHLDA3, BBC3, AEN, MDM2).

### M4: microbiome (M4-DRIFT-v1)

Each sample's distance from the same person's pre-flight samples (CLR abundances) is divided by that person's own pre-flight spread. **All four crew stayed within their own pre-flight range in flight** and were back near baseline by R+82. An exploratory pre- vs in-flight classifier reached AUROC 0.987 but dropped to 0.632 once the top 10 in-flight-only taxa were removed, so it is not used as a prediction model.

### M9: daily vital signs (M9-FLAG-v1, M9-MULTI-v1, M9-Q01)

```
modified z = 0.6745 × (today − median of previous 30 days) ÷ MAD
flag when |modified z| ≥ 3.5 and at least 21 valid baseline days exist
```

The modified z-score (Iglewicz & Hoaglin 1993) is robust to outliers. SpO2 is flagged but never alerts. Over the mission the rules raise **14 flags** (7 alerts, 7 SpO2 flags), identical to the M9 notebook output.

### M7: immune and inflammatory monitoring (M7-DEV-v1)

Per panel: z = (value − own pre-flight mean) ÷ (pooled within-person pre-flight SD × √(1 + 1/n)). The threshold k is the 95th percentile of leave-one-out pre-flight |z|, so about 5% of pre-flight samples would be flagged (measured 5.0–5.6%). A crew × timepoint counts as changed when more analytes are flagged than expected by chance (one-sided binomial p < 0.05). The Eve cytokine panel at R+82 was above chance for **C001 (13/71)** and **C002 (12/71)**, matching the team notebook. Because many correlated comparisons are made, "above chance" is descriptive, not a significance claim.

### M6: recovery after landing (M6-TRAJ-v2)

Every analyte outside its own pre-flight variation at R+1 is followed to R+45, R+82 and R+194. Across the four crew, **46 analytes changed at R+1 and 34 of 46 (74%) were back within own variation by R+82**.

### M8: telomeres

Only group-level results are published: telomeres were longer in flight in all four crew and shorter after return in three of four. M8 shows these statements as context and draws no per-person chart.

---

## Rules and recommendations

Rules run in priority order (1 = data quality and validity, 2 = status, 3 = routine). Every recommendation shows its rule ID and reason, and a test checks every recommendation for forbidden words (diagnosis, disease, medication, treatment and similar). All texts and thresholds live in `backend/data/reference/rules_v1.json`.

| Rule | Priority | Fires when | Recommendation |
|---|---|---|---|
| R02 | 1 | M3 request outside validated window or no own baseline | Outside validated analysis window; do not interpret a model output |
| R03 | 1 | M9-Q01 active (insufficient data) | Collect another measurement |
| R09 | 2 | M9-FLAG-v1 fires on an alert-enabled vital | Check the sensor and repeat; review with flight surgeon if it persists |
| R08 | 2 | M9-MULTI-v1 fires (2+ vitals changed) | Repeat all measurements and review with the flight surgeon |
| R10 | 2 | M1 alert in last 48 h, or M2 dose rate above background | Review exposure, follow the storm-shelter protocol |
| R11 | 2 | M3 reading elevated inside a validated window | Review the blood-response result with the flight surgeon |
| R12 | 2 | Physical (M2) and biological (M3) dose disagree | Repeat the blood assessment at the next validated window |
| R20 | 2 | M7 panel above chance | Continue immune monitoring; review mission medical protocol |
| R30 | 3 | M4 card available | Continue longitudinal microbiome monitoring |
| R40 | 3 | M8 context card | Continue longitudinal telomere monitoring |
| R50 | 2 | 2+ self-report items worse than own first 7 entries | Repeat the self-assessment; review behavioural-health procedures |
| R60 | 2 | >30 days in microgravity and no bone scan | Obtain a bone measurement per mission protocol |
| R70 | 3 | M6 result available | Continue longitudinal recovery monitoring |
| R99 | 3 | No other status rule active | Continue scheduled monitoring |

## Provenance: seven status types

Every displayed value carries one of these labels, so predictions and measurements can never be confused.

| Status type | Meaning | Examples |
|---|---|---|
| `MEASURED` | Directly measured | Bone scan entry, reaction time, Inspiration4 blood counts |
| `DERIVED` | Computed from measured values | M4 distance, M7/M6 flags |
| `ML_PREDICTED` | Trained model output | M1 SEP probability, M3 dose |
| `ESTIMATED` | Physics or approximation | M2 dose rate and mission dose |
| `CONTEXT` | Timeline or published parameter | Mission phase, M5, M8, bone context |
| `SELF_REPORTED` | Entered by the astronaut | Daily self-check |
| `TELEMETRY` | Crew vitals and cabin telemetry | M9 vital signs, cabin CO2 |

---

## Pages of the console

| Page | What it shows |
|---|---|
| **Home** | 3D astronaut-over-Earth scene with four live indicators; nine module cards; the six-step pipeline from data to action |
| **Crew Health Console (M9)** | Mission-day slider with play / jump-to-event; vital tiles with modified z and own baseline; live heart-rate stream; daily charts with own 30-day range and cabin CO2; flag feed and explanation card |
| **Radiation (M1 + M2)** | Real M1 test results; "Try the model" for any flare class (e.g. X1.0); replay of the 21 test storms; mission SEP probability, dose rate and cumulative dose by location |
| **Blood (M3)** | Choose a blood profile and hours since exposure; window indicator; dose class or X-ray-equivalent dose, or "No output"; M2 vs M3 comparison; full validation section; run the model on your own JSON sample |
| **Microbiome (M4)** | Status, in-flight distance, after-landing value and confidence; all-crew range bars; Tier A sites; contamination check |
| **Immune (M7)** | Analytes changed, timepoints above chance; flagged / tested / expected / p table; heat-map of largest changes; blood counts vs own range |
| **Telomere (M8)** | Before → in flight → after return timeline; comparison with the NASA Twins Study |
| **Recovery (M6)** | Indicator status table; R+1 changes back by R+82 for the crew member and all crew |
| **Body & Mind** | Mission context tiles; bone DXA panel; daily self-check sliders; five-trial reaction-time test; M5 brain evidence card |
| **Actions** | All recommendations for the selected crew member, sorted by priority, with rule ID, reason, module, status type and evidence link |
| **Evidence** | The seven status types, the 14-entry evidence registry and every provenance record for the selected subject |
| **About** | Architecture, evidence gate by domain and link to the API docs |

---

## API

All pages read from the local backend. There are **33 endpoints**; interactive documentation is at `/docs` when FastAPI is installed. All are `GET` unless marked.

| Endpoint | Returns |
|---|---|
| `/api/health`, `/api/meta`, `/api/crew` | Status, app info, mission crew and Inspiration4 crew |
| `/api/mission/{crew_id}/context?day=` | Phase, gravity, isolation, cabin CO2 |
| `/api/vitals/rules`, `/api/vitals/{crew_id}?day=` | Rule table; daily series, baseline bands, flags, data-quality notes |
| `/api/vitals/{crew_id}/flags`, `/api/vitals/{crew_id}/live` | Flag cards; live heart-rate samples |
| `/api/radiation/m1/summary`, `/storms`, `/timeline?around=` | M1 metrics, 21 test storms, replay |
| `POST /api/radiation/m1/predict` | Live M1 forecast, e.g. `{"flare_class": "X1.0"}` |
| `/api/radiation/forecast?day=`, `/api/radiation/{crew_id}/dose?day=&location=` | Mission M1 and M2 |
| `/api/blood/info`, `POST /api/blood/predict` | M3 validation data; prediction (demo profile or your own JSON) |
| `/api/microbiome/{crew_id}`, `/api/immune/{crew_id}`, `/api/telomere/{crew_id}`, `/api/recovery/{crew_id}`, `/api/brain` | Inspiration4 modules and M5 context |
| `/api/i4/calibration` | M7 / M6 thresholds per panel |
| `/api/selfreport/{crew_id}` (GET/POST), `POST /api/reaction/{crew_id}`, `/api/bone/{crew_id}` (GET/POST) | Body & Mind |
| `/api/actions/{subject_id}?day=` | Recommendations with rule ID and reason |
| `/api/indicators/{subject_id}?day=` | All provenance records |
| `/api/evidence`, `/api/evidence/{evidence_id}`, `/api/rules`, `/api/about` | Evidence registry, rule set, evidence gate |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/radiation/m1/predict \
     -H "Content-Type: application/json" \
     -d '{"flare_class": "X1.0"}'
```

---

## Testing

```bash
python -m unittest discover -s backend/tests -v
```

The 26 automated tests check that:

- M1 reproduces all 1,720 notebook test probabilities and the 14 caught / 7 missed storm counts;
- the M3 port gives exactly the same output as the team's original numpy predictor, and refuses outside validated windows, without a baseline or with missing genes;
- M9 reproduces exactly the notebook's 14 flags, deterministically, with SpO2 never alerting and no flag without 21 baseline days;
- the Inspiration4 data files are unchanged (checksums), complete for all crew and timepoints, free of race-coded fields, and NULISA zeros are treated as missing;
- M7 thresholds give 5% ± 1% pre-flight false flags and agree with the notebook; M6 counts are consistent; M8 is context only;
- every recommendation has a rule ID and reason and no forbidden wording; every displayed value has a known status type; every API route responds and bad input is rejected.

The test comparing M3 against the original numpy predictor is skipped if `numpy` is not installed. In addition, every page was opened for every crew member in a headless browser to check for script errors and empty or broken values.

---

## Limitations

- Only four Inspiration4 crew and three pre-flight samples per person; results are descriptive.
- M1 has few storms (wide confidence intervals) and many false alerts; probabilities are over-stated about three times on recent data.
- M3 is trained on ex vivo irradiated Earth blood with 5–11 donors per model; the 1 h signal was not reproduced in OSD-71; neutron readings are X-ray-equivalent.
- M2's storm coefficient and shielding factors still need calibration against MSL/RAD storm data.
- The urine inflammation panel (OSD-656) is not included; the Inspiration4 lab files were obtained from public mirrors and cross-checked rather than downloaded directly from OSDR.
- Mission telemetry for SIM-A to SIM-D is a mission scenario reconstructed from the M9 notebook, not data from a flown crew.

## Future work

- Calibrate M2 against MSL/RAD measurements during real solar particle events.
- Add the OSD-656 urine panel to M7 and connect real wearable streams to M9.
- Validate M3 on additional independent datasets and on in-flight blood if it becomes available.
- Test the console with flight-surgeon users and extend M1 with real-time DONKI and GOES feeds when a link to Earth exists.

---

## Team

**Team AstroZen**

| Name | Role |
|---|---|
| Faria Islam Sara | AI & Health Systems Lead |
| Habibur Rahman Hasib | Space Biology & Biomedical Research Lead |
| Mohammad Talha | Space Weather & Radiation Intelligence Lead |
| Rakibul Hasan | Software & Platform Engineering Lead |

---

## References

1. Garcia-Medina JS, Sienkiewicz K, Narayanan SA, et al., Mason CE. Genome and clonal hematopoiesis stability contrasts with immune, cfDNA, mitochondrial, and telomere length changes during short duration spaceflight. *Precision Clinical Medicine* 2024;7(1):pbae007. doi:10.1093/pcmedi/pbae007
2. Jones CW, et al. Molecular and physiological changes in the SpaceX Inspiration4 civilian crew. *Nature* 2024. doi:10.1038/s41586-024-07648-x
3. Kim J, Tierney BT, Overbey EG, et al. Single-cell multi-ome and immune profiles of the Inspiration4 crew reveal conserved, cell-type, and sex-specific responses to spaceflight. *Nature Communications* 2024;15. doi:10.1038/s41467-024-49211-2
4. Overbey EG, et al. Collection of biospecimens from the Inspiration4 mission establishes the standards for the Space Omics and Medical Atlas (SOMA). *Nature Communications* 2024. doi:10.1038/s41467-024-48806-z
5. Inspiration4 data access through the NASA Open Science Data Repository. *npj Microgravity* 2024;10:56. doi:10.1038/s41526-024-00393-5
6. Zeitlin C, Hassler DM, Cucinotta FA, et al. Measurements of energetic particle radiation in transit to Mars on the Mars Science Laboratory. *Science* 2013;340:1080–1084. doi:10.1126/science.1235989
7. Iglewicz B, Hoaglin DC. *How to Detect and Handle Outliers.* ASQC Quality Press, 1993.
8. NASA Human Research Program. Evidence Report: Risk of Cardiac Rhythm Problems During Spaceflight. NTRS 20170005625.
9. NASA-STD-3001 Volume 1: Crew Health (2022), career effective-dose limit 600 mSv.
10. NASA skeletal recovery study of long-duration crew, NTRS 20070001978.
11. NASA OSDR studies OSD-71, OSD-374, OSD-410, OSD-479, OSD-569, OSD-572, OSD-575, OSD-611, OSD-1012. https://osdr.nasa.gov
12. NASA DONKI space weather database, https://kauai.ccmc.gsfc.nasa.gov/DONKI/; NOAA GOES proton flux, https://www.swpc.noaa.gov

---

> **Disclaimer:** AstroCare is a research and demonstration prototype built for the NASA Space Apps Challenge 2026. It is **not a medical device and not for clinical use**. It never diagnoses, names a disease or recommends a medication.
