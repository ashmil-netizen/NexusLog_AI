# NexusLog AI

**Stage-Aware, Calibrated Early Warning for Delivery Delays: A Leakage-Audited Study on Public E-Commerce Data**

NexusLog AI is a retrospective research project for estimating whether an eligible Olist order was delivered after its estimated delivery date. Its three independently trained prediction stages simulate information available at order placement, payment approval, and carrier handover. It is not a live carrier-tracking service.

## Research question

How does delivery-delay prediction quality and probability calibration change as verified order-lifecycle information becomes available?

## Architecture

- `backend/scripts/`: download, inspect, prepare, and train workflows.
- `backend/app/`: FastAPI endpoints, schemas, and model/data services.
- `backend/data/`: KaggleHub cache files are copied here only when needed; generated data is ignored by Git.
- `backend/models/` and `backend/reports/`: trained artifacts and measured experiment outputs.
- `frontend/`: Vite, React, Tailwind CSS, React Router, and Recharts application.
- `research/`: methodology, experimental record, and limitations.

The actual delivery timestamp is used only to define the label. It is excluded from every model feature set. Item and payment tables are aggregated by order before joining. Temporal train/validation/test periods are separated in chronological order; validation is used for model selection, probability calibration, and threshold selection.

## Requirements and Windows setup

- Python 3.11 or later (the current environment was checked with Python 3.14).
- Node.js 20 or later and npm.
- Internet access for the KaggleHub dataset download and npm package installation.
- Kaggle account access may be required by KaggleHub. Do not put Kaggle credentials in this repository. Follow KaggleHub's own authentication guidance if it reports an authorization error.

In PowerShell, from the project root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_dataset.py
python scripts/inspect_dataset.py
python scripts/prepare_dataset.py
python scripts/train_model.py
```

If PowerShell blocks activation, use the Python executable directly (`.\.venv\Scripts\python.exe`) or follow your organization's execution-policy guidance; do not weaken machine-wide security settings just for this project.

Start the API in the activated backend terminal:

```powershell
uvicorn app.main:app --reload
```

In a second terminal:

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Open the Vite URL printed in the terminal. FastAPI's interactive API documentation is at `http://127.0.0.1:8000/docs`.

## Frontend configuration

`frontend/.env.example` sets `VITE_API_BASE_URL=http://127.0.0.1:8000/api`. This variable is not a secret. Restart Vite after changing it.

## Tests

```powershell
cd backend
python -m pytest
```

```powershell
cd frontend
npm run build
```

## API

- `GET /api/health`
- `GET /api/summary`
- `GET /api/orders?page=1&page_size=25&q=&late_delivery=&customer_state=&sort_by=order_purchase_timestamp&sort_order=desc`
- `GET /api/orders/high-risk?stage=A&limit=6`
- `GET /api/orders/{order_id}`
- `GET /api/orders/{order_id}/risk?stage=A`
- `POST /api/predict` with `{"order_id":"...","stage":"A"}`
- `GET /api/metrics`
- `GET /api/metrics/stages`
- `GET /api/metrics/calibration`
- `GET /api/metrics/cost-analysis`
- `GET /api/metrics/feature-importance`
- `GET /api/metrics/risk-distribution?stage=A`

The prediction endpoints score historical rows using only the selected stage's declared features. Orders without the relevant payment-approval or carrier-handover milestone are not eligible at that stage. Missing model artifacts produce a clear service error rather than a placeholder prediction.

Risk categories use the validation-selected threshold for that stage: **high** at or above the threshold, **medium** from half the threshold up to (but not including) it, and **low** below half. Historical ground truth is returned separately from risk predictions.

## Expected outputs

`prepare_dataset.py` creates `backend/data/processed/orders_processed.csv` and an audit report. `train_model.py` writes one model bundle per stage under `backend/models/` and real, machine-readable evaluation outputs under `backend/reports/`. The reports and downloaded/generated data are intentionally excluded from source control.

## Limitations

This is a retrospective simulation using public historical data, not a live delivery feed. Feature availability is inferred from dataset timestamps and should be validated against a real operating process before deployment. Costs used to choose alert thresholds are configurable illustrative assumptions. See [research/limitations.md](research/limitations.md).
