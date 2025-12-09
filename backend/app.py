import os
from flask import Flask, request, jsonify, render_template
import joblib
from flask_cors import CORS

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# ---- FIXED: Relative Path ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "backend", "model", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "backend", "model", "cv.pkl")

# Load Model & Vectorizer
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model Loaded Successfully")
except Exception as e:
    print(f"Error: {e}")
    model = None
    vectorizer = None

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    input_vec = vectorizer.transform([text]).toarray()
    prediction = model.predict(input_vec)[0]

    return jsonify({"language": prediction})

# ---- IMPORTANT FOR RENDER ----
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
