#color adjustments

import cv2

img = cv2.imread("C:/Users/Lenovo/Desktop/smart-photo-editor/data_sample_image/mertens.jpg")

resized_img = cv2.resize(img, (400, 400))

cv2.imshow("ilk resim", resized_img)


#grayscale çevirip kaydediyorum
imgGray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)


cv2.imshow("Grayscale img", imgGray)


cv2.imwrite("C:/Users/Lenovo/Desktop/smart-photo-editor/results_processed_images/grayscale.jpg", imgGray)  #grayscale halini kaydettim

cv2.waitKey(5000)
cv2.destroyAllWindows()

#parlaklık
deger = 50
parlaklik_artmis_img = cv2.add(imgGray, deger)
parlaklik_azalmis_img = cv2.subtract(imgGray, deger)

cv2.imshow("original grayscale", imgGray)
cv2.imshow("Parlaklik artmış grayscale ", parlaklik_artmis_img)
cv2.imshow("Parlaklik azalmis grayscale ", parlaklik_azalmis_img)

saved1 = cv2.imwrite("C:/Users/Lenovo/Desktop/smart-photo-editor/results_processed_images/parlaklik_artmis_img.jpg", parlaklik_artmis_img)
saved2 = cv2.imwrite("C:/Users/Lenovo/Desktop/smart-photo-editor/results_processed_images/parlaklik_azalmis_img.jpg", parlaklik_azalmis_img)

if saved1 and saved2:
    print(" Görseller başarıyla kaydedildi.")
else:
    print("Kaydetme işlemi başarısız oldu!")


cv2.waitKey(5000)
cv2.destroyAllWindows()