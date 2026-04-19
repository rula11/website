from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import gdown

app = Flask(__name__)

MODEL_PATH = "model.onnx"
DATA_PATH = "model.onnx.data"

# ======================
# DOWNLOAD MODEL (AUTO)
# ======================
if not os.path.exists(MODEL_PATH):
    print("Download ONNX...")
    gdown.download(
        "https://drive.google.com/uc?id=17Fa1tk2AiGwRtKbaEPDSNhAkJw8MZ7lg",
        MODEL_PATH,
        quiet=False
    )

if not os.path.exists(DATA_PATH):
    print("Download DATA...")
    gdown.download(
        "https://drive.google.com/uc?id=1wuCtmbUzxG3D4QdRXKmVAE6yLlLs6U7_",
        DATA_PATH,
        quiet=False
    )

# load model
session = ort.InferenceSession(MODEL_PATH)

# ======================
# PREPROCESS
# ======================
def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return image

# ======================
# ROUTE
# ======================
@app.route("/")
def home():
    return "API ONNX jalan 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        image = Image.open(file.stream).convert("RGB")

        input_data = preprocess(image)

        outputs = session.run(None, {"input": input_data})
        prob = 1 / (1 + np.exp(-outputs[0][0][0]))

        label = "High" if prob > 0.5 else "Normal"

        return jsonify({
            "prediction": label,
            "confidence": float(prob)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":
    app.run()
