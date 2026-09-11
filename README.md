# Real-time Payment Fraud Detection

This repository implements a compact, production-shaped fraud detection portfolio:

- Phase 1: PaySim ingestion, cleaning, and Great Expectations data contract.
- Phase 2: chronological splits, time-window aggregates, and graph features.
- Phase 3: rules baseline, cost-sensitive LightGBM, MLflow, and operations metrics.
- Phase 4: FastAPI `/predict`, latency measurement, and PSI drift utilities.
- Phase 5: Streamlit investigator queue and Docker Compose deployment.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python scripts/download_paysim.py --output data/processed/paysim_clean.csv
python -m src.data.validate data/processed/paysim_clean.csv
python -m src.models.train data/processed/paysim_clean.csv
pytest
```

Start the local services after training:

```bash
docker compose up --build
```

Then open `http://localhost:8501` for the dashboard or `http://localhost:8000/docs`
for the API. Without an artifact, the API intentionally falls back to the rules
baseline. The source dataset is synthetic PaySim data; never place private banking
data in this repository.
