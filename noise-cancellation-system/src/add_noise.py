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
print(img.dtype)       # float mı, uint8 mi?
print(np.min(img))     # En düşük piksel değeri kaç?
print(np.max(img))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(img), plt.title("Original Image"), plt.show()

gaussianNoisyImage = gaussianNoise(img)
plt.figure(), plt.imshow(gaussianNoisyImage), plt.axis("off"), plt.title("Gaussian Noise Image"), plt.show()
gaussianNoisyImage1 = cv2.cvtColor(gaussianNoisyImage, cv2.COLOR_RGB2BGR)
cv2.imwrite("gaussianNoisyImage.jpg",gaussianNoisyImage1)

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
plt.figure(), plt.imshow(spImage), plt.axis("off"), plt.title("Salt Pepper Image"), plt.show()

spImage1 = cv2.cvtColor(spImage, cv2.COLOR_RGB2BGR)
cv2.imwrite("SaltPepperImage.jpg",spImage1)





