# Heart Risk MLOps — Level 1 Auto Train Pipeline

An end-to-end MLOps Level 1 project implementing automated training,
experiment tracking, batch deployment, drift detection, and CI.

## Architecture
```
Data Source → Automated ETL → ML Pipeline → MLflow Tracking
     → Model Registry → Batch Inference → Drift Detection → Retrain
```

## Stack
- **ML:** scikit-learn, pandas, numpy
- **Tracking:** MLflow (experiment tracking + model registry)
- **Containerization:** Docker
- **Testing:** pytest
- **CI:** GitHub Actions

## Project Structure
```
heart-risk-mlops/
├── config/config.yaml     # All hyperparameters live here
├── src/
│   ├── ingest.py          # Automated ETL
│   ├── preprocess.py      # Feature pipeline
│   ├── train.py           # MLflow-tracked training
│   ├── evaluate.py        # Metrics + plots
│   ├── batch_infer.py     # Batch inference
│   ├── data_drift.py      # PSI drift detection
│   └── compare_models.py  # Auto-promotion logic
├── tests/                 # pytest unit tests
├── docker/                # Train + inference Dockerfiles
└── .github/workflows/     # CI pipeline
```

## Quickstart
```bash
# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run full pipeline
python src/ingest.py
python src/train.py
python src/batch_infer.py

# Run retraining pipeline
retrain.bat

# Launch MLflow UI
mlflow ui --backend-store-uri mlruns

# Run tests
pytest tests/ -v
```

## Results
- AUC-ROC: **0.9491** on UCI Heart Disease dataset
- 303 patients, 13 features, binary classification
- Full experiment history tracked in MLflow

## Model Card
See [model_card.md](model_card.md) for dataset details, metrics, and limitations.