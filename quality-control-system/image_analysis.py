import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), 0)
        if img is not None:
            img = cv2.resize(img,(200,200))
            images.append(img)
    return images

flaws = load_images_from_folder("flaws/")
imperfects = load_images_from_folder("imperfects/")