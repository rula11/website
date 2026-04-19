from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

# ======================
# LOAD MODEL
# ======================
model = models.resnet50()

model.fc = nn.Sequential(
    nn.Linear(2048, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 1)
)

import os
import gdown

MODEL_PATH = "A1-Resnet50.pth"

if not os.path.exists(MODEL_PATH):
    print("Download model dulu...")
    gdown.download("https://drive.google.com/uc?id=1KVIVbUwpnlDHcphG7uENc_g6uHnOjhFW", MODEL_PATH, quiet=False)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ======================
# TRANSFORM (SAMA SEPERTI TRAINING)
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# ======================
# ROUTE
# ======================
@app.route("/")
def home():
    return "API jalan 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]

        # baca gambar
        image = Image.open(file.stream).convert("RGB")
        image = transform(image).unsqueeze(0)

        # prediksi
        with torch.no_grad():
            output = model(image)
            prob = torch.sigmoid(output).item()

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
