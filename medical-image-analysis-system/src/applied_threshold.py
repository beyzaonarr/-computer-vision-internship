import cv2

img = cv2.imread("rontgen.jpg", 0)
resized_img = cv2.resize(img,(600,600))
blur_img = cv2.GaussianBlur(resized_img,(5,5),0)


ret, thotsu_img = cv2.threshold(blur_img, 0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow("Original Image with Gaussian Blur", blur_img)
cv2.imshow("Threshold Image", thotsu_img)
cv2.imwrite("threshold.jpg", thotsu_img)
cv2.waitKey(0)

adaptive_threshold_image = cv2.adaptiveThreshold(blur_img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize = 21, C = 5)
cv2.imshow("Adaptive Threshold Image", adaptive_threshold_image)
cv2.imwrite("adaptive_threshold_meanC.jpg", adaptive_threshold_image)

cv2.waitKey(0)

adaptive_threshold_image2 = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize = 21, C = 5)
cv2.imshow("Adaptive Threshold Image Gaussian-C Method", adaptive_threshold_image2)
cv2.imwrite("adaptive_threshold_gaussianC.jpg", adaptive_threshold_image2)

cv2.waitKey(0)
cv2.destroyAllWindows()