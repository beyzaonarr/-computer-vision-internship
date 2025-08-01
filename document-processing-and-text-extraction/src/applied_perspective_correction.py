import cv2
import numpy as np

img = cv2.imread("cap2.png")
resize_img = cv2.resize(img,(500,500))

orig = resize_img.copy()

gray_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

edges = cv2.Canny(blur_img, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key = cv2.contourArea, reverse = True)

for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4:
        document_corners = approx
        break


def order_points(pts):
    # Noktaları sırala: [sol üst, sağ üst, sağ alt, sol alt]
    rect = np.zeros((4, 2), dtype="float32")
    pts = pts.reshape(4, 2)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # sol üst
    rect[2] = pts[np.argmax(s)]  # sağ alt

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # sağ üst
    rect[3] = pts[np.argmax(diff)]  # sol alt

    return rect

pts1 = order_points(document_corners)

width, height = 400, 600
pts2 = np.float32([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
])

matrix = cv2.getPerspectiveTransform(pts1, pts2)

warped = cv2.warpPerspective(orig, matrix, (width, height))

cv2.imshow("Original", resize_img)
cv2.imshow("Corrected Perspective", warped)
cv2.imwrite("perspective.png",warped)
cv2.waitKey(0)
cv2.destroyAllWindows()










