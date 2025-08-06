import cv2

healthy_leaf = cv2.imread("healthy_leaf.webp")
cv2.imshow("healthy leaf", healthy_leaf)
cv2.waitKey(0)

diseases_leaf = cv2.imread("diseased_leaf.jpg")
cv2.imshow("diseases leaf", diseases_leaf)
cv2.waitKey(0)
cv2.destroyAllWindows()