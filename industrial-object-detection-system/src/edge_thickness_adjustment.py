import cv2
import numpy as np

img1 = cv2.imread("home.jpg", 0)
img = cv2.resize(img1,(500,500))
blur_img = cv2.GaussianBlur(img,(3,3),0)

canny_image = cv2.Canny(blur_img, 100, 200)
kernel = np.ones((3,3),np.uint8)
dilates_edges = cv2.dilate(canny_image, kernel, iterations=1)

cv2.imshow("Canny Image with Dilates Edges", dilates_edges)
cv2.imwrite("canny_edges_dilate_edges.jpg", dilates_edges)
cv2.waitKey(3000)
cv2.destroyAllWindows()

erode_edges = cv2.erode(canny_image, kernel, iterations=1)

cv2.imshow("Canny Image with Eroded Edges", erode_edges)
cv2.imwrite("canny_edges_erode_edges.jpg", erode_edges)
cv2.waitKey(3000)
cv2.destroyAllWindows()

sobelx = cv2.Sobel(blur_img, cv2.CV_64F,1,0, 5)
sobely = cv2.Sobel(blur_img, cv2.CV_64F,0,1,5)

sobelx_abs = cv2.convertScaleAbs(sobelx)
sobely_abs = cv2.convertScaleAbs(sobely)
sobel_combined = cv2.addWeighted(sobelx_abs, 0.5, sobely_abs,0.5,0)

sobel_dilate_edges = cv2.dilate(sobel_combined,kernel,1)

cv2.imshow("Sobel Image with Dilates Edges", sobel_dilate_edges)
cv2.imwrite("sobel_dilate_edges.jpg", sobel_dilate_edges)
cv2.waitKey(3000)
cv2.destroyAllWindows()

sobel_erode_edges = cv2.erode(sobel_combined,kernel,1)

cv2.imshow("Sobel Image with Eroded Edges", sobel_erode_edges)
cv2.imwrite("sobel_erode_edges.jpg", sobel_erode_edges)
cv2.waitKey(3000)
cv2.destroyAllWindows()

