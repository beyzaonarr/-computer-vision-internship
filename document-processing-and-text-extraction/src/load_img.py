import cv2

img = cv2.imread("cap2.png")
print(img.shape)

cv2.imshow("orijinal", img)
cv2.waitKey(0)

resize_img = cv2.resize(img,(500,500))
cv2.imshow("resized image", resize_img)
cv2.imwrite("resized_image.png", resize_img)

cv2.waitKey(0)
cv2.destroyAllWindows()