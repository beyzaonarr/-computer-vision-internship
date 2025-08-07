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
imperfects = load_images_from_folder("imperfects/")

template = np.mean(flaws, axis = 0).astype(np.uint8)

test_img = imperfects[50]
roi = test_img[50:150, 50:150]
template_roi = template[50:150, 50:150]

res = cv2.matchTemplate(roi, template_roi, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

threshold = 0.85

if max_val < threshold:
    print("Flaw detected!")

    h, w = template_roi.shape
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)


    cv2.rectangle(test_img, top_left, bottom_right, (0, 0, 255), 2)

    cv2.imshow(" Flaw Detected", test_img)
    cv2.imwrite("flaw_detected.jpg", test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("The product is perfect")


