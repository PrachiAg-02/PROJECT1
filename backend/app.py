import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Target labels for the Iris dataset
TARGET_NAMES = ["Setosa", "Versicolor", "Virginica"]

# Load trained model
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Loaded model successfully.")
else:
    print(f"Warning: Model not found at {MODEL_PATH}")

@app.route("/")
def home():
    return {"message": "Flask ML Server is running!"}

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    features = data.get("features")

    if not features:
        return jsonify({"error": "No 'features' provided in request body"}), 400

    try:
        features_array = np.array([float(x) for x in features]).reshape(1, -1)
        pred_idx = int(model.predict(features_array)[0])
        class_name = TARGET_NAMES[pred_idx] if pred_idx < len(TARGET_NAMES) else "Unknown"

        return jsonify({
            "status": "success",
            "prediction": pred_idx,
            "class_name": class_name
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)