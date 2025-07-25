import cv2

img1 = cv2.imread("home.jpg", 0)
img = cv2.resize(img1,(500,500))

blur_img = cv2.GaussianBlur(img,(3,3),0)
cv2.imshow("blur img", blur_img)
cv2.imwrite("blur_img.jpg", blur_img)
cv2.waitKey(3000)
cv2.destroyAllWindows()

canny_image = cv2.Canny(blur_img, 100, 200)
cv2.imshow("canny image", canny_image)
cv2.imwrite("canny_image.jpg", canny_image)
cv2.waitKey(3000)
cv2.destroyAllWindows()

sobelx = cv2.Sobel(blur_img, cv2.CV_64F,1,0, 5)
sobely = cv2.Sobel(blur_img, cv2.CV_64F,0,1,5)

sobelx_abs = cv2.convertScaleAbs(sobelx)
sobely_abs = cv2.convertScaleAbs(sobely)

sobel_combined = cv2.addWeighted(sobelx_abs, 0.5, sobely_abs,0.5,0)
cv2.imwrite("sobel_combined.jpg", sobel_combined)
cv2.imwrite("sobelx.jpg", sobelx_abs)
cv2.imwrite("sobely.jpg", sobely_abs)


cv2.imshow("sobel x image", sobelx_abs)
cv2.imshow("sobel y image", sobely_abs)
cv2.imshow("sobel combined image", sobel_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

