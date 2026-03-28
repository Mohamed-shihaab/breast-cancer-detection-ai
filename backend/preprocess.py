import cv2
import numpy as np


def preprocess_image(image):
    """
    Input: BGR image (numpy array)
    Output: Preprocessed grayscale image
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize (standard size)
    gray = cv2.resize(gray, (256, 256))

    # Gaussian blur (noise removal)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Normalize
    norm = blur / 255.0

    return norm
