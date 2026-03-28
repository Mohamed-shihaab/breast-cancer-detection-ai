import cv2
from preprocess import preprocess_image

output = preprocess_image("sample.jpg")

cv2.imshow("Preprocessed Image", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
