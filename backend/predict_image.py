import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

MODEL_PATH = "cnn_transfer_finetuned.keras"
IMG_SIZE = 224

model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        print(f"🧪 Prediction: MALIGNANT ({pred:.2f})")
    else:
        print(f"🧪 Prediction: BENIGN ({1 - pred:.2f})")

# 🔁 CHANGE IMAGE PATH HERE
predict_image("sample.jpg")
