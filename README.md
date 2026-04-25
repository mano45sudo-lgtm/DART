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

## 4) End-to-end training (env rollouts, not a static dataset)

The training loop lives in `scripts/train_reinforce_twin.py`. Each REINFORCE update calls `rollout_episode` → `DigitalTwinDiabetesEnv.reset` / `step` with a live LM policy, so gradients come from **on-policy trajectories in the simulator**, not from a fixed offline dataset. Baselines are a **random policy** (and an **untrained LM** line on the learning-curve figure) evaluated on the same env.

**Two run modes**

| Mode | Command | Role |
|------|---------|------|
| Smoke (fast CPU check) | `python scripts/train_reinforce_twin.py --quick` | CI / layout verification; uses `sshleifer/tiny-gpt2` (may not parse valid JSON actions). |
| Judge / demo (meaningful length) | `python scripts/train_reinforce_twin.py --judge-preset` | **Recommended for reviewers:** `distilgpt2`, 120 updates, 4 episodes/update, 32 held-out eval seeds, 80 random baseline episodes. Use a GPU if you can. |

After any run, **re-commit** `logs/training_last.json` and the two PNGs under `docs/figures/` so judges see your latest numbers and plots. Optional one-liner to **stage** those paths for `git commit` (run from a git checkout):

```bash
python scripts/train_reinforce_twin.py --judge-preset --git-stage-artifacts
```

Figures are written as **high-resolution PNG** (160 DPI) with **labeled axes** (update index; episode return in the env’s reward units).

### Plots (committed in-repo)

**Learning curve + baselines** (`docs/figures/training_vs_baselines.png`):

![Training curve vs random and untrained LM baselines](docs/figures/training_vs_baselines.png)

**Final bar comparison** (held-out seeds; `docs/figures/final_random_vs_trained.png`):

![Final eval random vs trained LM](docs/figures/final_random_vs_trained.png)

**Random baseline only** (`logs/baseline_returns.png` from `scripts/plot_rewards.py`):

![Random agent returns per episode](logs/baseline_returns.png)

### Latest logged metrics (from committed `logs/training_last.json`)

These numbers update whenever you re-run training and overwrite the JSON. *Current commit (smoke `--quick` run):*

| Metric | Random baseline | Untrained LM (eval@0) | Trained LM (final held-out) |
|--------|-----------------|------------------------|-----------------------------|
| Mean episode return | 6.45 (std 30.81, *n*=20 eval episodes) | −24.85 (std 21.81, *n*=8) | −24.85 (std 21.81, *n*=8) |
| JSON action parse rate | — | 0.00 | 0.00 |

The smoke model does not emit parseable tool JSON reliably; **run `--judge-preset` with `distilgpt2`**, then refresh this table from the new `training_last.json` before submission.

### Notebook (Colab)

- `training/train_trl_colab.ipynb` — installs deps, runs smoke or judge training, displays the same PNGs.

### Weights & Biases (optional)

If you log a run to Wandb, add the **direct URL to that run** here (plots may still be mirrored in-repo as above):

- `<ADD_WANDB_RUN_URL_HERE>`

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

## 7) Google Colab (end-to-end env training)

Open:
- `training/train_trl_colab.ipynb`

Colab steps:
1. **File → Open notebook → GitHub** (paste your repo URL) and open the notebook.
2. Run install cell(s).
3. Set `REPO_ROOT` in the notebook (e.g., `/content/mano/DART`).
4. Run the quick smoke cell once:
   - `python scripts/train_reinforce_twin.py --quick`
5. For submission-quality curves, run the **judge preset** (GPU recommended); equivalent to the long manual flags:
   - `python scripts/train_reinforce_twin.py --judge-preset`
   - Optional: `python scripts/train_reinforce_twin.py --judge-preset --git-stage-artifacts` then `git commit` from the repo root.
6. **Commit** these paths so reviewers see plots and numbers in GitHub (not only inside Colab):
   - `logs/training_last.json`
   - `docs/figures/training_vs_baselines.png`
   - `docs/figures/final_random_vs_trained.png`

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

