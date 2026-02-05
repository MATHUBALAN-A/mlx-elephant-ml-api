
import os
import pandas as pd
import pickle # Changed from joblib
from flask import Flask, request, jsonify

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

# Feature columns (as defined in the training process)
feature_columns = [f"px_{i}" for i in range(768)]

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

        # Expecting: list of 768 values (single sample) or list of lists (multiple samples)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            # Multiple samples
            input_df = pd.DataFrame(data, columns=feature_columns)
        elif isinstance(data, list) and len(data) == 768:
            # Single sample
            input_df = pd.DataFrame([data], columns=feature_columns)
        else:
            return jsonify({"error": "Invalid input format. Expected a list of 768 pixel values or a list of such lists."}), 400

        preds = model.predict(input_df)
        result = [get_label_name(int(p)) for p in preds]

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ================= RUN SERVER =================

if __name__ == "__main__":
    # Render automatically sets the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
