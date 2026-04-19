from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image

app = Flask(__name__)

# ======================
# LOAD MODEL (ONNX)
# ======================
MODEL_PATH = "model.onnx"

session = ort.InferenceSession(MODEL_PATH)

# ======================
# PREPROCESS (SAMAKAN DENGAN TRAINING)
# ======================
def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0

    # Normalize (sama seperti training kamu)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    # HWC → CHW
    image = np.transpose(image, (2, 0, 1))

    # tambah batch dimension
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

        # baca gambar
        image = Image.open(file.stream).convert("RGB")
        input_data = preprocess(image)

        # inference ONNX
        outputs = session.run(None, {"input": input_data})

        # sigmoid manual (karena output masih logit)
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

# ======================
# RUN
# ======================
if __name__ == "__main__":
    app.run(debug=True)
