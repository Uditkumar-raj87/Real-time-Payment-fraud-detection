# Real-time Payment Fraud Detection

This repository implements a compact, production-shaped fraud detection portfolio:


## Quickstart

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,validation]'
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
The source dataset is synthetic PaySim data; never place private banking data
in this repository.
