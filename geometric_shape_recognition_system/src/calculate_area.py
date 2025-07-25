import cv2
from matplotlib import pyplot as plt

img = cv2.imread("img.webp")
img_copy = img.copy()

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized_image = cv2.resize(gray_img,(900,700))

img_gaussian = cv2.GaussianBlur(resized_image,(5,5),0)

image = cv2.Canny(img_gaussian,100,200)

contours, hierarchy = cv2.findContours(image,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img_copy,contours, -1,(0,255,0),4)

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour,True)

    print(f"Countur {i+1}:")
    print(f"Area (pixels): {area:.2f}") #virgülden sonraki iki basamak için 2f
    print(f"Perimeter (pixels): {perimeter:.2f}")
