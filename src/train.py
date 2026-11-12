
import pandas as pd
import yaml
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# Load parameters
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

# Load training dataset
train_data = pd.read_csv("data/processed/train.csv")

# Separate features and target
X_train = train_data.drop("target", axis=1)
y_train = train_data["target"]

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=params["random_forest"]["n_estimators"],
    max_depth=params["random_forest"]["max_depth"],
    random_state=params["random_state"]
)

# Train model
model.fit(X_train, y_train)

# Create models directory
Path("models").mkdir(exist_ok=True)

# Save trained model
joblib.dump(model, "models/model.pkl")

print("Model training completed successfully!")
print("Model saved at models/model.pkl")
