import cv2
import numpy as np


img = cv2.imread("rontgen.jpg")
resized_img = cv2.resize(img,(600,600))
img_copy1 = np.copy(resized_img)
img_copy2 = np.copy(resized_img)

gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

blur_img = cv2.GaussianBlur(gray_img,(3,3),0)

ret, thotsu_img = cv2.threshold(blur_img, 0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3,3), np.uint8)

otsu_cleaned = cv2.morphologyEx(thotsu_img, cv2.MORPH_OPEN,kernel)
otsu_cleaned = cv2.morphologyEx(otsu_cleaned, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(otsu_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 100:
        cv2.drawContours(img_copy1, [cnt], -1, (0,0,255),2)

cv2.imshow("contours threshold", img_copy1)
cv2.imwrite("contours_threshold.jpg", img_copy1)
cv2.waitKey(0)



adaptive_threshold_image = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize = 21, C = 5)

adaptive_cleaned =  cv2.morphologyEx(adaptive_threshold_image, cv2.MORPH_OPEN, kernel)
adaptive_cleaned = cv2.morphologyEx(adaptive_cleaned, cv2.MORPH_CLOSE, kernel)

contours_adaptive, _ = cv2.findContours(adaptive_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt_a in contours_adaptive:
    area = cv2.contourArea(cnt_a)
    if area > 100:
        cv2.drawContours(img_copy2, [cnt], -1, (0,0,255), 2)

cv2.imshow("contours adaptive", img_copy2)
cv2.imwrite("contours_thresh_adaptive.jpg", img_copy2)
cv2.waitKey(0)
cv2.destroyAllWindows()