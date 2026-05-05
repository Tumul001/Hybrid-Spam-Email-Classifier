# NLP_Novalantis — Codebase & Agent Chat Overview

## What This Project Is

A **spam vs not-spam email classifier** built incrementally through a GitHub Copilot agent conversation. It has evolved from a single training script into a **full-stack modular system**:

- **ML Domain Layer** (`src/spam_detector/`) — core NLP logic
- **FastAPI Backend** (`backend/`) — REST API exposing the ML functions
- **Streamlit Frontend** (`streamlit_app.py`) — paste-and-check web UI
- **CLI Scripts** (`scripts/`) — terminal wrappers for training, evaluation, prediction
- **Tests** (`tests/`) — unit + smoke + API integration tests

---

## Project Structure at a Glance

```
NLP_novalantis/
├── src/spam_detector/          ← ML Domain Layer
│   ├── preprocess.py           ← text cleaning + label normalization
│   ├── model.py                ← TF-IDF + Logistic Regression pipeline
│   ├── train.py                ← training orchestration + metrics saving
│   ├── evaluate.py             ← evaluation + confusion matrix
│   ├── predict.py              ← single + batch prediction, threshold logic
│   ├── threshold.py            ← auto-tune review threshold to a target rate
│   └── io_utils.py             ← save/load model + JSON helpers
│
├── backend/                    ← FastAPI REST API
│   ├── main.py                 ← FastAPI app + route definitions
│   ├── service.py              ← calls spam_detector functions
│   ├── schemas.py              ← Pydantic request/response schemas
│   └── config.py               ← paths/defaults config
│
├── streamlit_app.py            ← Streamlit UI (calls backend via HTTP)
├── scripts/                    ← CLI entry points
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── tune_threshold.py
├── tests/
│   ├── test_preprocess.py
│   ├── test_smoke_workflow.py
│   ├── test_threshold.py
│   └── api/test_backend_api.py
├── data/
│   └── raw/spam.csv            ← your real dataset (Category, Message columns)
├── models/                     ← saved artifacts (spam_model.joblib, metrics.json)
└── requirements.txt
```

---

## How the Data Flows

```
spam.csv (Category, Message)
  ↓  preprocess.py: auto-detects columns, cleans text, normalizes labels
  ↓  model.py: TF-IDF (unigram+bigram) + LogisticRegression
  ↓  save → spam_model.joblib + metrics.json

Streamlit UI  →  FastAPI (/api/v1/predict/text)
              →  service.py  →  predict.py  →  spam_model.joblib
              ←  { prediction, confidence, needs_review, review_threshold }
```

---

## The ML Pipeline (Core Logic)

| Step | What Happens |
|---|---|
| **Column auto-detection** | Recognizes `Message`/`text`/`body` and `Category`/`label`/`class` automatically |
| **Text cleaning** | Strips HTML, URLs, emails, punctuation → lowercase |
| **Label normalization** | Maps `ham`→`not spam`, `spam`→`spam`, `0/1` supported |
| **TF-IDF** | `max_features=8000`, unigram+bigram, English stop words |
| **Logistic Regression** | `class_weight="balanced"`, `max_iter=400` |
| **Confidence scoring** | `predict_proba` per prediction |
| **Review threshold** | If confidence < threshold → flagged for manual review |
| **Auto-tuning** | `tune_threshold.py` finds threshold that achieves ~N% review rate |

**Threshold resolution priority:**
1. Manual `--review-threshold` override
2. Auto-loaded from `models/review_threshold.json` (if it exists)
3. Default fallback: `0.6`

---

## FastAPI Backend Endpoints

| Method | Endpoint | What it does |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/api/v1/train` | Train model on a CSV file |
| POST | `/api/v1/evaluate` | Evaluate model, return metrics |
| POST | `/api/v1/predict/text` | Classify a single email text |
| POST | `/api/v1/predict/batch` | Classify all rows in a CSV |
| POST | `/api/v1/threshold/tune` | Find the best threshold for a target review rate |

> **Why FastAPI?** Separates the ML logic from the UI. The same backend can be called from Streamlit, a mobile app, a Chrome extension, or any HTTP client. Pydantic validates all inputs automatically.

---

## Streamlit UI (Frontend)

The UI (`streamlit_app.py`) **no longer imports ML code directly** — it sends HTTP requests to the FastAPI backend.

Sections:
- **Sidebar** — API URL, model path, threshold config, optional manual threshold slider
- **Single Email** — text area → Check Email → shows SPAM/NOT SPAM + metrics
- **Batch CSV** — CSV path input → Run Batch Prediction
- **Training & Metrics** — Train / Evaluate / Tune Threshold buttons

---

## Your Dataset (`data/raw/spam.csv`)

- **Columns:** `Category` (spam/ham), `Message`
- **Auto-detected** by `preprocess.py` — no renaming needed
- Was previously: `sample_emails.csv` (AI-generated, now deleted)
- This is the same dataset used in the **Demo Proj notebook** (`email-spam-detection-98-accuracy.ipynb`)

---

## Demo Project vs Our Project

| | Demo Notebook | Our Project |
|---|---|---|
| **Model** | CountVectorizer + MultinomialNB | TF-IDF + LogisticRegression |
| **Accuracy on spam.csv** | ~98% | ~95% (more generalizable) |
| **Structure** | Single `.ipynb` notebook | Modular backend + frontend |
| **API** | None | FastAPI REST |
| **UI** | None | Streamlit |
| **Model saving** | No | Yes (`spam_model.joblib`) |
| **Threshold logic** | No | Yes (auto-tune + review flags) |
| **Tests** | No | Yes (10 passing tests) |
| **Column flexibility** | Hardcoded | Auto-detects column names |

**Recommendation from agent:** Keep TF-IDF + LR as default (better generalization); NB can be added as a "demo profile" option later.

---

## What the Agent Chat Shows

The `agent chat.txt` is the full VS Code GitHub Copilot agent session log (~3200 lines) showing:

1. **Phase 1:** Initial spam NLP setup — simple TF-IDF + LR script
2. **Phase 2:** Workspace scaffold — full project structure created
3. **Phase 3:** Confidence scoring & review threshold logic added
4. **Phase 4:** Auto-threshold tuning module added (`threshold.py`)
5. **Phase 5:** Threshold auto-loading in prediction (no manual entry needed)
6. **Phase 6:** Streamlit UI added for paste-and-check
7. **Phase 7:** FastAPI backend + Streamlit refactored to call API
8. **Phase 8:** Column mapping for `spam.csv` (`Category/Message` auto-detected)
9. **Final user question:** "Will removing FastAPI make things simpler? But I need a normal backend"

---

## Current Status & What's Incomplete

> [!IMPORTANT]
> The agent hit a **rate limit** at the end of the chat. The last user message asked:
> *"will removing fastapi and rest make things simpler? but i need a normal backend"*
> The agent responded but was cut off — **this is where you currently are**.

> [!WARNING]
> The column mapping for `spam.csv` was implemented in `preprocess.py` (it auto-detects `Category`/`Message`), but tests in `test_smoke_workflow.py` and `test_threshold.py` may still reference `sample_emails.csv` (old AI-generated data). You need to run `pytest` to verify.

---

## How to Run Right Now

### 1. Activate virtualenv
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Train the model on your real data
```powershell
python scripts/train.py --data data/raw/spam.csv
```

### 3. Start the backend
```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Start the frontend (new terminal)
```powershell
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser → paste email → click **Check Email**

### 5. Run tests
```powershell
pytest
```

---

## Open Questions / Next Steps

1. **Do you want to replace FastAPI with a simpler backend?** (e.g., Flask, or even no API — direct Python imports in Streamlit)
2. **Do you want to add the NB model as an alternative** to compare accuracy vs your TF-IDF baseline?
3. **Tests may be broken** since `sample_emails.csv` was deleted — run `pytest` and share output if you want fixes.
4. **Batch CSV upload in Streamlit** (file upload widget + download button) — agent offered this but it wasn't implemented.
