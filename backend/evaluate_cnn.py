import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

IMG_SIZE = 224
BATCH_SIZE = 32

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dataset")
)

model = tf.keras.models.load_model("cnn_transfer_finetuned.keras")

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    BASE_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

preds = model.predict(val_gen)
y_pred = (preds > 0.5).astype(int).ravel()
y_true = val_gen.classes

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Benign", "Malignant"]))
