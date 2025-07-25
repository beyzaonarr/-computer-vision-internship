import cv2

img = cv2.imread("home.jpg", 0)
print(img.shape)

cv2.imwrite("gray_home.jpg", img)

cv2.imshow("Orijinal Gray Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()