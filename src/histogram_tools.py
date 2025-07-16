#histogram tools
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("C:/Users/Lenovo/Desktop/smart-photo-editor/data_sample_image/mertens.jpg")
resized_img = cv2.resize(img, (400, 400))
cv2.imshow("ilk resim", resized_img)


imgGray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale img", imgGray)

cv2.waitKey(5000)

#histogram hesaplıyorum
img_hist = cv2.calcHist([imgGray], [0], None, [256], [0, 256])
print(img_hist.shape)

#histogram çiz
plt.figure(figsize=(8, 8))
plt.title("Grayscale Histogram")
plt.xlabel("Piksel Değeri")
plt.ylabel("Piksel Sayısı")
plt.plot(img_hist, color = "red")
plt.grid()
plt.show()