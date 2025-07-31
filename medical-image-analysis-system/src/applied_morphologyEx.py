import cv2
import numpy as np


img = cv2.imread("rontgen.jpg", 0)
resized_img = cv2.resize(img,(600,600))
blur_img = cv2.GaussianBlur(resized_img,(5,5),0)

ret, thotsu_img = cv2.threshold(blur_img, 0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3,3), np.uint8)

otsu_cleaned = cv2.morphologyEx(thotsu_img, cv2.MORPH_OPEN,kernel)
otsu_cleaned = cv2.morphologyEx(otsu_cleaned, cv2.MORPH_CLOSE, kernel)

cv2.imshow("otsu threshold cleaned", otsu_cleaned)
cv2.imwrite("otsu_threshold_cleaned.jpg", otsu_cleaned)
cv2.waitKey(0)

adaptive_threshold_image = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize = 21, C = 5)

adaptive_cleaned =  cv2.morphologyEx(adaptive_threshold_image, cv2.MORPH_OPEN, kernel)
adaptive_cleaned = cv2.morphologyEx(adaptive_cleaned, cv2.MORPH_CLOSE, kernel)

cv2.imshow("adaptive threshold cleaned", adaptive_cleaned)
cv2.imwrite("adaptive_threshold_cleaned.jpg", adaptive_cleaned)
cv2.waitKey(0)
cv2.destroyAllWindows()
