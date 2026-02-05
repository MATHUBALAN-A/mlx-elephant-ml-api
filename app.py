
import os
# import pandas as pd # Removed pandas
import pickle
from flask import Flask, request, jsonify
import numpy as np # Added numpy

app = Flask(__name__)

# ================= LOAD MODEL =================
# On Render, app.py and xgb_model.pkl are expected to be in the same directory
MODEL_PATH = "xgb_model.pkl"

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Feature columns are no longer explicitly needed if passing NumPy arrays directly
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

        # Convert to NumPy array
        # Expecting: list of 768 values (single sample) or list of lists (multiple samples)
        if not isinstance(data, list):
            return jsonify({"error": "Invalid input format. Expected a list or list of lists."}), 400

        # Validate input structure and convert to numpy array
        if len(data) == 0:
            return jsonify({"error": "Input list cannot be empty."}), 400

        # If it's a list of lists, treat as multiple samples
        if isinstance(data[0], list):
            input_array = np.array(data, dtype=float)
            if input_array.shape[1] != 768:
                return jsonify({"error": f"Each sample must contain exactly 768 pixel values. Received {input_array.shape[1]}."}), 400
        # If it's a single list, treat as single sample
        elif isinstance(data, list) and len(data) == 768:
            input_array = np.array([data], dtype=float) # Wrap in a list for consistent 2D shape
        else:
            return jsonify({"error": "Invalid input format. Expected a list of 768 pixel values or a list of lists of 768 pixel values."
            }), 400


        # Make predictions using the NumPy array directly
        preds = model.predict(input_array)
        result = [get_label_name(int(p)) for p in preds]

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ================= RUN SERVER =================

if __name__ == "__main__":
    # Render automatically sets the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
