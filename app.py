
import os
# Removed pandas as per user constraint
import joblib # Changed from pickle to joblib
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# ================= LOAD MODEL =================
MODEL_PATH = "xgb_model.joblib" # Changed to .joblib

try:
    model = joblib.load(MODEL_PATH) # Changed to joblib.load
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Feature columns no longer needed as model will be trained on NumPy arrays without names
# feature_columns = [f"px_{i}" for i in range(768)]

# Label mapping
def get_label_name(label):
    if label == 0:
        return "Empty"
    elif label == 1:
        return "Human"
    elif label == 2:
        return "Elephant"
    else:
        return "Unknown"

# ================= ROUTES =================

@app.route("/")
def home():
    return "🐘 XGBoost Elephant Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True)

        # Convert to NumPy array directly, validate shape
        if not isinstance(data, list):
            return jsonify({"error": "Invalid input format. Expected a JSON array."}), 400

        if len(data) == 0:
            return jsonify({"error": "Input JSON array cannot be empty."}), 400

        # Determine if single or multiple samples and convert to NumPy array
        if isinstance(data[0], list):
            # Multiple samples
            input_array = np.array(data, dtype=float)
            if input_array.shape[1] != 768:
                return jsonify({"error": f"Each sample must contain exactly 768 pixel values. Received {input_array.shape[1]}."}), 400
        elif isinstance(data, (int, float)) or (isinstance(data, list) and len(data) == 768 and all(isinstance(x, (int, float)) for x in data)):
            # Single sample (list of 768 values)
            input_array = np.array([data], dtype=float) # Wrap in a list for consistent 2D shape
        else:
            return jsonify({"error": "Invalid input format. Expected a list of 768 pixel values or a list of lists of 768 pixel values."}), 400

        # Make predictions using the NumPy array directly
        preds = model.predict(input_array)
        result = [get_label_name(int(p)) for p in preds]

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ================= RUN SERVER =================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
