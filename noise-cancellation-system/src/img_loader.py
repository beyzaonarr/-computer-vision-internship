import cv2

img = cv2.imread("remus.jpg")
cv2.imshow("Original Image", img)
cv2.waitKey(2000)
cv2.destroyAllWindows()

print(img.shape)

resized_img = cv2.resize(img, (700,500))
cv2.imshow("Resized Image", resized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()