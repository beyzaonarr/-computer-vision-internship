import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("home.jpg", 0)
img = cv2.resize(img1,(500,500))

blur_img = cv2.GaussianBlur(img,(3,3),0)

canny_image = cv2.Canny(blur_img, 100, 200)

canny_image_sobelx = cv2.Sobel(canny_image,cv2.CV_64F,1,0,5)
canny_image_sobely = cv2.Sobel(canny_image,cv2.CV_64F,0,1,5)

magnitude_canny_image = cv2.magnitude(canny_image_sobelx,canny_image_sobelx)
print(magnitude_canny_image)

magnitude_canny_image_uint8 = cv2.convertScaleAbs(magnitude_canny_image)
cv2.imwrite("magnitude_gradient_canny_image.jpg", magnitude_canny_image_uint8)

cv2.imshow("magnitude gradient canny image", magnitude_canny_image_uint8)
cv2.waitKey(3000)
cv2.destroyAllWindows()
 

sobel_image_sobelx = cv2.Sobel(blur_img,cv2.CV_64F,1,0,5)
sobel_image_sobely = cv2.Sobel(blur_img,cv2.CV_64F,0,1,5)

magnitude_sobel_image = cv2.magnitude(sobel_image_sobelx,sobel_image_sobely)
print(magnitude_sobel_image)

magnitude_sobel_image_uint8 = cv2.convertScaleAbs(magnitude_sobel_image)
cv2.imwrite("magnitude_gradient_sobel_image.jpg", magnitude_sobel_image_uint8)

cv2.imshow("magnitude gradient sobel image", magnitude_sobel_image_uint8)
cv2.waitKey(3000)
cv2.destroyAllWindows()

plt.figure(), plt.imshow(magnitude_sobel_image_uint8, cmap = "gray") ,plt.title("Magnitude Sobel Image"), plt.show()
plt.figure(), plt.imshow(magnitude_canny_image_uint8, cmap = "gray") ,plt.title("Magnitude Canny Image"), plt.show()