import cv2
import numpy as np

def gabor_features(image):
    image = image.astype(np.float32)

    features = []

    for theta in np.arange(0, np.pi, np.pi / 16):
        kernel = cv2.getGaborKernel(
            ksize=(31, 31),
            sigma=4.0,
            theta=theta,
            lambd=10.0,
            gamma=0.5,
            psi=0,
            ktype=cv2.CV_32F
        )

        filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
        features.append(filtered.mean())

    return np.array(features)
