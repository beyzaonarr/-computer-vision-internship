import cv2
import numpy as np


def calculate_psnr(original, noisy):
    mse = np.mean((original - noisy) ** 2)
    if mse == 0:
        return float('inf') #sonsuz= görüntüler aynıysa

    max_pixel = 255
    psnr = 10 * np.log10(max_pixel / mse)
    return psnr


def gaussianNoise(img):
    row, col, ch = img.shape
    mean = 0
    var = 0.05 #standart sapma elde etmek için
    sigma = var ** 0.5

    gauss = np.random.normal(mean, sigma, (row,col,ch)) * 255
    gauss = gauss.reshape(row,col,ch)

    noisy = img.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8) #görseller uint 8 formatında okutulur ve kaydedilir unutma

img = cv2.imread("remus.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gaussianNoisyImage = gaussianNoise(img)

def saltPepperNoise(img):
    row, col, ch = img.shape
    s_vs_p = 0.5

    amount = 0.007

    spnimg = np.copy(img)

    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i, int(num_salt)) for i in img.shape]
    spnimg[tuple(coords)] =  255

    num_pepper = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i, int(num_pepper)) for i in img.shape]
    spnimg[tuple(coords)] = 0

    return spnimg

spImage = saltPepperNoise(img)


mbGaussNoisyImage = cv2.medianBlur(gaussianNoisyImage,5)
psnr1 = calculate_psnr(img, mbGaussNoisyImage)
print("Gauss gürültüsü eklenmiş image'ye medianblur filtresi uyguladığımızda oluşan sonuç ve original image'miz arasındaki PSNR değeri:  ", psnr1, "dB")

mbsaltpepperNoisyImage = cv2.medianBlur(spImage,5)
psnr2 = calculate_psnr(img, mbsaltpepperNoisyImage)
print("Salt Pepper gürültüsü eklenmiş image'ye medianblur filtresi uyguladığımızda oluşan sonuç ve original image'miz arasındaki PSNR değeri: ", psnr2, "dB")

gaussianBlurGaussNoisyImage = cv2.GaussianBlur(gaussianNoisyImage,(5,5),0)
psnr3 = calculate_psnr(img, gaussianBlurGaussNoisyImage)
print("Gauss gürültüsü eklenmiş image'ye GaussianBlur filtresi uyguladığımızda oluşan sonuç ve original image'miz arasındaki PSNR değeri: ", psnr3,"dB")

gaussianBlurSaltPepperImage = cv2.GaussianBlur(spImage,(5,5), 0)
psnr4 = calculate_psnr(img, gaussianBlurSaltPepperImage)
print("Salt Pepper gürültüsü eklenmiş image'ye  GaussianBlur filtresi uyguladığımızda oluşan sonuç ve original image'miz arasındaki PSNR değeri:", psnr4, "dB")

bilateralFilterGaussNoisyImage = cv2.bilateralFilter(gaussianNoisyImage,5,75,75)
psnr5 = calculate_psnr(img, bilateralFilterGaussNoisyImage)
print("Gauss gürültüsü eklenmiş image'ye bilateral filtresi uyguladığımızda oluşan sonuç ve original image'miz arasındaki PSNR değeri: ", psnr5, "dB")

bilateral_filter_salt_pepper_image = cv2.bilateralFilter(spImage,5,75,75)
psnr6 = calculate_psnr(img,bilateral_filter_salt_pepper_image)
print("Salt Pepper gürültüsü eklenmiş image'ye bilateral filtresi uyguladığımızda oluşan sonuç ve original image'miz arasındaki PSNR değeri:", psnr6, "dB")

