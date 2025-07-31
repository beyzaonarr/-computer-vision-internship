import cv2

img = cv2.imread("cap2.png")
resize_img = cv2.resize(img,(500,500))

orig = resize_img.copy()

gray_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

edges = cv2.Canny(blur_img, 50, 150)
cv2.imshow("edges", edges)
cv2.waitKey(3000)


contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key = cv2.contourArea, reverse = True)

for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4:
        document_corners = approx
        break

for point in document_corners:
    cv2.circle(orig, tuple(point[0]), 10, (0, 255, 0), -1)

cv2.imshow("corners", orig)
cv2.imwrite("corners.jpg", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()