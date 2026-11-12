
import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

# Load parameters
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

# Load DVC-tracked raw dataset
df = pd.read_csv("data/raw/iris.csv")

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Create processed data directory
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["test_size"],
    random_state=params["random_state"],
    stratify=y
)

# Combine features and target
train_data = X_train.copy()
train_data["target"] = y_train

test_data = X_test.copy()
test_data["target"] = y_test

# Save datasets
train_data.to_csv("data/processed/train.csv", index=False)
test_data.to_csv("data/processed/test.csv", index=False)

print("Dataset preparation completed!")
print("Training records:", len(train_data))
print("Testing records:", len(test_data))
