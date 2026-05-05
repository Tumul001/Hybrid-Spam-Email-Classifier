# Spam Email NLP Classifier

This project now uses a modular full-stack setup:
- Backend API: FastAPI
- Frontend: Streamlit
- Domain ML modules: `src/spam_detector/`

## Install

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run backend

```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

## Run frontend

```powershell
streamlit run streamlit_app.py
```

Set API URL in Streamlit sidebar if needed (default `http://127.0.0.1:8000`).

## API endpoints (v1)

- `GET /health`
- `POST /api/v1/train`
- `POST /api/v1/evaluate`
- `POST /api/v1/predict/text`
- `POST /api/v1/predict/batch`
- `POST /api/v1/threshold/tune`

## Legacy CLI scripts

These still work:

```powershell
python scripts/train.py --data data/raw/spam.csv
python scripts/evaluate.py --data data/raw/spam.csv
python scripts/predict.py --text "You won a free gift card. Click now!"
python scripts/tune_threshold.py --data data/raw/spam.csv --target-review-rate 0.20
```

## Tests

```powershell
pytest
```

Includes domain tests and API integration tests in `tests/api/`.

