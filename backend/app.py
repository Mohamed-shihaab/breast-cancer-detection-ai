from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
CORS(app)

# Load CNN model
MODEL_PATH = "cnn_transfer_finetuned.keras"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.70  # 🔥 important

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Breast Cancer Detection API Running"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    temp_path = "temp.jpg"
    file.save(temp_path)

    img = preprocess_image(temp_path)
    prediction = model.predict(img)[0][0]

    confidence = float(max(prediction, 1 - prediction))

    # 🚫 Reject non-histopathology images
    if confidence < CONFIDENCE_THRESHOLD:
        os.remove(temp_path)
        return jsonify({
            "prediction": "Invalid Image",
            "confidence": round(confidence * 100, 2),
            "message": "Please upload a histopathology image"
        })

    label = "Malignant" if prediction > 0.5 else "Benign"

    os.remove(temp_path)

    return jsonify({
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
