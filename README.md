# Real-time Payment Fraud Detection

An end-to-end portfolio project for detecting fraudulent synthetic mobile-money
transactions from the PaySim dataset. It demonstrates data contracts,
leakage-safe features, cost-sensitive modeling, API serving, drift monitoring,
and an investigator dashboard.

## Project Contents

- `data_contracts/`: Great Expectations contract for the cleaned PaySim schema.
- `src/data/`: PaySim download, cleaning, and contract validation.
- `src/features/`: Chronological splits, time-window features, and graph features.
- `src/models/`: Rules baseline, cost-sensitive LightGBM, MLflow metrics, and evaluation.
- `src/monitoring/`: Population Stability Index drift calculations.
- `api/`: FastAPI `/predict` and `/health` endpoints.
- `dashboard/`: Streamlit risk-operations dashboard and investigator queue.
- `notebooks/01_eda.ipynb`: Basic exploratory analysis.
- `tests/`: Unit tests for ingestion, features, splits, and drift monitoring.
- `Dockerfile` and `docker-compose.yml`: Local container deployment.

The PaySim dataset is synthetic. Do not add private banking or payment data to
this repository.

## 1. Check Prerequisites

Use Python 3.12 for the complete setup. Great Expectations 0.18 requires a
NumPy version that does not currently install cleanly on Python 3.14.

```bash
python3.12 --version
git --version
```

Use Python 3.10, 3.11, 3.12, or 3.13 if Python 3.12 is not installed.

## 2. Create the Environment

Run these commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,validation]'
```

If `ensurepip` is unavailable, use `virtualenv`:

```bash
python3.12 -m pip install --user --break-system-packages virtualenv
python3.12 -m virtualenv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,validation]'
```

The `dev` extra installs testing and linting tools. The `validation` extra
installs Great Expectations and PyArrow support.

## 3. Download and Clean PaySim

```bash
python scripts/download_paysim.py \
	--output data/processed/paysim_clean.csv
```

The downloader fetches the public synthetic CSV, selects required columns,
normalizes transaction types, converts numeric fields, and writes the cleaned
file under `data/processed/`. That directory is ignored by Git.

To use a different public mirror:

```bash
python scripts/download_paysim.py \
	--url 'https://example.com/paysim.csv' \
	--output data/processed/paysim_clean.csv
```

## 4. Validate the Data Contract

```bash
python -m src.data.validate data/processed/paysim_clean.csv
```

The contract checks ordered columns, nullability, allowed transaction types,
fraud-label domains, flagged-fraud domains, and non-negative numeric values.
A valid dataset exits with status `0`; a failed contract exits with status `1`.

Parquet is also supported:

```bash
python scripts/download_paysim.py --output data/processed/paysim_clean.parquet
python -m src.data.validate data/processed/paysim_clean.parquet
```

## 5. Train the Model

```bash
python -m src.models.train \
	data/processed/paysim_clean.csv \
	--artifact artifacts/model.joblib
```

Training uses chronological PaySim steps instead of random splitting:

- 60% earliest steps for training.
- 20% following steps for validation.
- Remaining latest steps for testing.

The model logs validation metrics to MLflow and writes the serving bundle to
`artifacts/model.joblib`. Metrics include PR-AUC, recall at review capacity,
Precision@K, expected fraud loss prevented, false-decline rate, and the review
threshold. Raw accuracy is not used as the primary metric.

## 6. Run Tests and Linting

```bash
pytest -q
ruff check .
```

## 7. Start the FastAPI Service

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

In another terminal, check health:

```bash
curl http://localhost:8000/health
```

Score a transaction:

```bash
curl -X POST http://localhost:8000/predict \
	-H 'content-type: application/json' \
	-d '{
		"step": 1,
		"type": "TRANSFER",
		"amount": 100,
		"nameOrig": "C1",
		"oldbalanceOrg": 100,
		"newbalanceOrig": 0,
		"nameDest": "M1",
		"oldbalanceDest": 0,
		"newbalanceDest": 100,
		"isFlaggedFraud": 0
	}'
```

Open interactive API documentation at `http://localhost:8000/docs`. If
`artifacts/model.joblib` does not exist, the API uses the rules baseline and
reports `"model": "rules"`.

## 8. Start the Streamlit Dashboard

Keep the API running. In a second terminal:

```bash
source .venv/bin/activate
streamlit run dashboard/app.py --server.port 8501
```

Open `http://localhost:8501`. Use the sidebar to score a transaction through
the API. Upload a cleaned PaySim CSV to view counts, fraud rate, transaction
rows, and the investigator queue ordered by amount.

For a different API address:

```bash
export FRAUD_API_URL=http://localhost:8000
streamlit run dashboard/app.py --server.port 8501
```

## 9. Run with Docker Compose

```bash
docker compose up --build
```

Open the dashboard at `http://localhost:8501`, the API at
`http://localhost:8000`, and API documentation at `http://localhost:8000/docs`.

Stop the services with:

```bash
docker compose down
```

The trained model is optional. Mount or create `artifacts/model.joblib` before
starting the API container to use LightGBM; otherwise the rules fallback keeps
the service operational.

## 10. Explore the Notebook

Open `notebooks/01_eda.ipynb` in VS Code or Jupyter after downloading the data.
It expects `data/processed/paysim_clean.csv` and displays sample transactions
and fraud rates by transaction type.

## Troubleshooting

### Great Expectations tries to compile NumPy

Use Python 3.10 through 3.13 and recreate the environment:

```bash
rm -rf .venv
python3.12 -m virtualenv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,validation]'
```

### A port is already in use

```bash
uvicorn api.main:app --port 8001
streamlit run dashboard/app.py --server.port 8502
```

Update `FRAUD_API_URL` if the API port changes.

### The dashboard says the API is unavailable

Start the API first, confirm `curl http://localhost:8000/health` works, and
check that `FRAUD_API_URL` points to the same host and port.
