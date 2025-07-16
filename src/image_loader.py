#image loader
import cv2

img = cv2.imread("C:/Users/Lenovo/Desktop/smart-photo-editor/data_sample_image/mertens.jpg")
print(img.shape)

resized_img = cv2.resize(img, (400, 400))

cv2.imshow("ilk resim", resized_img)

cv2.waitKey(0)
