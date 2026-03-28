import os
import cv2
import numpy as np
from preprocess import preprocess_image
from gabor import extract_gabor_features



def extract_from_folder(folder_path, label):
    X = []
    y = []

    print(f"\nScanning folder: {folder_path}")

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(root, file)

                try:
                    # Preprocess image
                    pre = preprocess_image(img_path)

                    # Extract features
                    features = extract_gabor_features(pre)

                    X.append(features)
                    y.append(label)

                except Exception as e:
                    print(f"⚠️ Skipped {img_path}: {e}")

    print(f"✔ Loaded {len(X)} images from {folder_path}")
    return np.array(X), np.array(y)


def main():
    # Absolute dataset path (BEST PRACTICE)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")

    benign_path = os.path.join(DATASET_DIR, "benign")
    malignant_path = os.path.join(DATASET_DIR, "malignant")

    print("Dataset path:", DATASET_DIR)

    # Extract features
    X_benign, y_benign = extract_from_folder(benign_path, 0)
    X_malignant, y_malignant = extract_from_folder(malignant_path, 1)

    # Combine
    if len(X_benign) == 0 and len(X_malignant) == 0:
        print("❌ No features extracted. Check dataset folders.")
        return

    X = np.vstack((X_benign, X_malignant))
    y = np.hstack((y_benign, y_malignant))

    print("\n✅ Feature extraction completed")
    print("Total samples:", X.shape[0])
    print("Feature length:", X.shape[1])

    # 🔥 SAVE FEATURES (THIS FIXES YOUR ERROR)
    np.save("X.npy", X)
    np.save("y.npy", y)

    print("✅ Features saved as X.npy and y.npy")


if __name__ == "__main__":
    main()
