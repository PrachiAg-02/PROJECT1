import os
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 1. Load sample dataset
data = load_iris()
X, y = data.data, data.target

# 2. Train a simple model
model = RandomForestClassifier()
model.fit(X, y)

# 3. Save the model to the 'models' directory
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, 'model.pkl')

joblib.dump(model, model_path)
print(f"Model saved successfully to: {model_path}")