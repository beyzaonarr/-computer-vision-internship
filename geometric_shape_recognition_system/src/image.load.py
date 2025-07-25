import cv2

gray_img = cv2.imread("img.webp", 0)
print(gray_img.shape)
resized_image = cv2.resize(gray_img,(900,700))

img_gaussian = cv2.GaussianBlur(resized_image,(5,5),0)

image = cv2.Canny(img_gaussian,50,150)


cv2.imshow("Image", image)
cv2.imwrite("used_image.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()