
import pandas as pd
import joblib
import json

from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load testing dataset
test_data = pd.read_csv("data/processed/test.csv")

# Separate features and target
X_test = test_data.drop("target", axis=1)
y_test = test_data["target"]

# Load trained model
model = joblib.load("models/model.pkl")

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

# Display results
print("Model Evaluation Results")
print("------------------------")
print("Accuracy:", accuracy)
print("F1 Score:", f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create reports directory
Path("reports").mkdir(exist_ok=True)

# Save metrics
metrics = {
    "accuracy": float(accuracy),
    "f1_score": float(f1)
}

with open("reports/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nMetrics saved to reports/metrics.json")
