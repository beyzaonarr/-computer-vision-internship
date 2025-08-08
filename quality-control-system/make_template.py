import cv2
import os
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), 0)
        if img is not None:
            img = cv2.resize(img,(200,200))
            images.append(img)
    return images

flaws = load_images_from_folder("flaws/")
flawless = load_images_from_folder("flawless/")

template = np.mean(flawless, axis = 0).astype(np.uint8)
cv2.imshow("template", template)
cv2.imwrite("template.jpg", template)
cv2.waitKey(0)
cv2.destroyAllWindows()
