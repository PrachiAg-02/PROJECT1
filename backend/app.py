import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
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
    features = data.get("features")  # Expecting a list like: [5.1, 3.5, 1.4, 0.2]

    if not features:
        return jsonify({"error": "No 'features' provided in request body"}), 400

    # Format input and make prediction
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)

    return jsonify({
        "status": "success",
        "prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)