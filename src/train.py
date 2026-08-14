import os
import json
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Paths Setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

for d in [DATASET_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# 2. Save / Load Raw Dataset
raw_data = load_iris(as_frame=True)
df = raw_data.frame
csv_path = os.path.join(DATASET_DIR, 'iris.csv')
df.to_csv(csv_path, index=False)
print(f"Dataset stored at: {csv_path}")

# 3. Train / Test Split
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate Model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=raw_data.target_names, output_dict=True)

# 6. Save Metrics to 'results/'
metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump({"test_accuracy": acc, "report": report}, f, indent=4)

# 7. Save Trained Model to 'models/'
model_path = os.path.join(MODELS_DIR, 'model.pkl')
joblib.dump(model, model_path)

print(f"Test Accuracy: {acc * 100:.2f}%")
print(f"Model saved to: {model_path}")
print(f"Metrics saved to: {metrics_path}")