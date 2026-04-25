---
title: Digital Twin Medicine — RL Agent for Personalized Treatment
emoji: "🧬"
colorFrom: indigo
colorTo: purple
sdk: streamlit
app_file: app.py
---

## Digital Twin Medicine — RL Agent for Personalized Treatment

- **Theme**: OpenEnv Hackathon 2026 — **#3.1 World Modeling (Professional Tasks)**
- **Core capability gap**: sequential treatment decisions for *an individual patient* under uncertainty (not a one-shot predictor).
- **Deliverable**: a **digital twin environment** + **tool-augmented RL agent loop** + **end-to-end training evidence**.

## Links (fill before submission)
- **UI Space (Streamlit)**: `<ADD_SPACE_URL_HERE>`
- **OpenEnv Env Space (Docker/FastAPI)**: `<ADD_OPENENV_SPACE_URL_HERE>`
- **Google Colab training**: `<ADD_COLAB_URL_HERE>`
- **Mini-blog / video (<2 min)**: `<ADD_BLOG_OR_VIDEO_URL_HERE>`

## 1) Problem → Environment → Agent (3-minute overview)

- **Problem**: “one-size-fits-many” medicine fails because the *same* drug sequence produces different outcomes across individuals.
- **Environment**: a **partially observable** T2DM patient digital twin; each step = **1 simulated week**; state evolves stochastically with treatment effects + side effects + costs.
- **Agent task**: output a **JSON treatment action** each week (start/add/stop/switch/dose_adjust + lifestyle).
- **Tools**: EHR, genomics, interactions, progression forecast, trial sim, biomarkers, resistance, risk (all implemented in `tools/`).
- **Reward**: dense clinical improvement + trend shaping, plus penalties (toxicity/cost) and terminal outcomes.

## 2) What the agent sees / does (OpenEnv-style)

### Observation (partial)
Returned every step from `env.reset()` / `env.step()`:
- `week`, `hba1c`, `fasting_glucose`, `bmi`, `systolic_bp`, `egfr`, `ckd`, `cvd`

### Action (JSON)
Examples:
```json
{"type":"start","drug":"metformin","dose":1.0,"lifestyle":0.7}
```
```json
{"type":"add","drug":"glp1","dose":1.0}
```

## 3) Reward (dense + hard-to-game)
Implemented in `reward/rubric.py`:
- **Clinical improvement**: HbA1c + fasting glucose reduction
- **Biomarker shaping**: distance-to-target + BMI trend
- **Side effects**: severity-weighted penalty
- **Cost**: weekly cost penalty
- **Terminal**: remission bonus / failure penalty

## 4) Results (baseline vs trained)

### Baseline (random agent)
Generated with:
```powershell
python scripts/run_evaluation.py
python scripts/plot_rewards.py
```

Plot (committed):
- `logs/baseline_returns.png` — episode return vs episode (random agent)

### Trained (HF TRL PPO in Colab)
Notebook:
- `training/train_trl_colab.ipynb`

You must export and commit:
- `logs/trained_returns.png` — **avg return vs training iteration**
- (optional) `logs/parse_ok_rate.png` — JSON action parse success vs iteration

## 5) Reproduce locally (Windows PowerShell)

```powershell
cd "C:\Users\DELL\Downloads\Digital_twin"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts/run_sanity.py
```

### Environment demo (single episode)
```powershell
python scripts/run_env_demo.py
```

### Baseline evaluation + plot (evidence)
```powershell
python scripts/run_evaluation.py
python scripts/plot_rewards.py
```

### UI (Streamlit dashboard)
```powershell
python scripts/run_ui.py
```
Open the printed URL (usually `http://localhost:8501`).

## 6) Reproduce locally (macOS/Linux)

```bash
cd Digital_twin
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts/run_sanity.py
```

## 7) Google Colab (HF TRL end-to-end training)

Open:
- `training/train_trl_colab.ipynb`

Colab steps:
1. **File → Open notebook → GitHub** (paste your repo URL) and open the notebook.
2. Run install cell(s).
3. Run training cells until reward curve stabilizes (don’t stop after 2–3 iterations).
4. Save plots as PNG and commit to `logs/`:
   - `logs/trained_returns.png`
   - (optional) `logs/parse_ok_rate.png`

## 8) Hugging Face Spaces (discoverable + runnable)

### Space A — UI (Streamlit)
- SDK: **Streamlit**
- Entry: root `app.py` (delegates to `ui/app.py`)

### Space B — OpenEnv Env Server (Docker/FastAPI) — required
- SDK: **Docker**
- **Repository subdirectory**: `spaces/openenv`
- Runs: `uvicorn dtm_openenv.server.app:app --port 7860`

Verify endpoints:
- `/health` → `{"ok": true}`
- `/docs` → interactive FastAPI docs

## 9) Engineering compliance checklist (judges’ table stakes)

- **Gym-style API**: `env.digital_twin_env.DigitalTwinDiabetesEnv` implements `reset`, `step`, `state` ✅
- **Valid `openenv.yaml`**: `spec_version: 1` + client/action/observation/state entries ✅
- **Client/server separation**: env server code lives under `dtm_openenv/server/` ✅
- **No reserved tool names used as tools**: tools are in `tools/*.py` (not named reset/step/state/close) ✅

```text
env/           # digital twin environment + patient twin
tools/         # tool system modules
reward/        # reward rubric
training/      # notebooks
evaluation/    # baseline + metrics
ui/            # Streamlit dashboard UI
dtm_openenv/   # OpenEnv-style server + models
scripts/       # entrypoints
logs/          # reward curves, run logs
spaces/openenv # Docker Space for env server
```

