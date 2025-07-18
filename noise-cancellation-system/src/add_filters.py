import cv2
import matplotlib.pyplot as plt
import numpy as np

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


#median blur ekliyorum gürültülü resimlere
mbGaussNoisyImage = cv2.medianBlur(gaussianNoisyImage,5)
plt.figure(), plt.imshow(mbGaussNoisyImage), plt.axis("off"), plt.title("MB Gaussian Noise Image"), plt.show()

mbGaussNoisyImage1 = cv2.cvtColor(mbGaussNoisyImage, cv2.COLOR_RGB2BGR )
cv2.imwrite("MedianBlurGaussNoise.jpg",mbGaussNoisyImage1)



mbsaltpepperNoisyImage = cv2.medianBlur(spImage,5)
plt.figure(), plt.imshow(mbsaltpepperNoisyImage), plt.axis("off"), plt.title("MB salt pepper Noise Image"), plt.show()

mbsaltpepperNoisyImage1 = cv2.cvtColor(mbsaltpepperNoisyImage, cv2.COLOR_RGB2BGR )
cv2.imwrite("MedianBlurSaltPepperImage.jpg", mbsaltpepperNoisyImage1)

#gaussian blur ekliyorum
gaussianBlurGaussNoisyImage = cv2.GaussianBlur(gaussianNoisyImage,(5,5),0)
plt.figure(), plt.imshow(gaussianBlurGaussNoisyImage), plt.axis("off"), plt.title("Gaussian Blur Gauss Noise Image"), plt.show()

gaussianBlurGaussNoisyImage1 = cv2.cvtColor(gaussianBlurGaussNoisyImage, cv2.COLOR_RGB2BGR)
cv2.imwrite("GaussianBlurGaussNoise.jpg", gaussianBlurGaussNoisyImage1)



gaussianBlurSaltPepperImage = cv2.GaussianBlur(spImage,(5,5), 0)
plt.figure(), plt.imshow(gaussianBlurSaltPepperImage), plt.axis("off"), plt.title("Gaussian Blur Salt Pepper Image"), plt.show()

gaussianBlurSaltPepperImage1 = cv2.cvtColor(gaussianBlurSaltPepperImage, cv2.COLOR_RGB2BGR)
cv2.imwrite("GaussianBlurSaltPepperImage.jpg", gaussianBlurSaltPepperImage1)

#bilateral filter
bilateralFilterGaussNoisyImage = cv2.bilateralFilter(gaussianNoisyImage,5,75,75)
plt.figure(), plt.imshow(bilateralFilterGaussNoisyImage), plt.axis("off"), plt.title("Bilateral Filter Gauss Noise Image"), plt.show()

bilateralFilterGaussNoisyImage1 = cv2.cvtColor(bilateralFilterGaussNoisyImage, cv2.COLOR_RGB2BGR)
cv2.imwrite("bilateralFilterGaussianNoiseImage.jpg", bilateralFilterGaussNoisyImage1)



bilateral_filter_salt_pepper_image = cv2.bilateralFilter(spImage,5,75,75)
plt.figure(), plt.imshow(bilateral_filter_salt_pepper_image), plt.axis("off"), plt.title("Bilateral Filter Salt Pepper Image"), plt.show()

bilateral_filter_salt_pepper_image1 = cv2.cvtColor(bilateral_filter_salt_pepper_image, cv2.COLOR_RGB2BGR)
cv2.imwrite("bilateral_filter_saltpepper_image.jpg", bilateral_filter_salt_pepper_image1)