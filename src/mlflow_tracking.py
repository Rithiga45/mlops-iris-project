
import mlflow
import mlflow.sklearn
import joblib
import json
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load trained model
model = joblib.load(BASE_DIR / "models" / "model.pkl")

# Load evaluation metrics
with open(BASE_DIR / "reports" / "metrics.json", "r") as file:
    metrics = json.load(file)

# Set MLflow experiment
mlflow.set_experiment("Iris Flower Classification")

with mlflow.start_run():
    # Log model parameters
    mlflow.log_param("model", "RandomForestClassifier")

    # Log evaluation metrics
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(name, value)

    # Log trained model
    mlflow.sklearn.log_model(
    model,
    name="iris_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)

    print("MLflow tracking completed!")
    print("Metrics:", metrics)
