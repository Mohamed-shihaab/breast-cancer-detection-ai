from preprocess import preprocess_image
from gabor import gabor_features
from pca_reduce import reduce_features

img = preprocess_image("sample.jpg")
gabor = gabor_features(img)
pca_feat = reduce_features(gabor)

print("Reduced feature length:", len(pca_feat))
print(pca_feat)
