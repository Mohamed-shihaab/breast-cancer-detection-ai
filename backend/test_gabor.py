from preprocess import preprocess_image
from gabor import gabor_features

img = preprocess_image("sample.jpg")
features = gabor_features(img)

print("Gabor feature vector length:", len(features))
print(features)
