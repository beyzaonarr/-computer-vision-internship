import cv2

img = cv2.imread("rontgen.jpg", 0)
print(img.shape)

resized_img = cv2.resize(img,(600,600))
cv2.imshow("resized_img", resized_img)
cv2.waitKey(0)

blur_img = cv2.GaussianBlur(resized_img,(5,5),0)

cv2.imshow("blur_img", blur_img)
cv2.imwrite("blur_img.jpg", blur_img)

cv2.waitKey(0)
cv2.destroyAllWindows()




