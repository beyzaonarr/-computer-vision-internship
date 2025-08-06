import cv2
import numpy as np

def extract_color_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Image is not load!")
        return None

    avg_color_per_row = np.average(image, axis = 0) #her bir sütun ortları
    avg_color = np.average(avg_color_per_row, axis = 0) # ortların otalaması

    return avg_color

healthy_leaf_average = extract_color_features("healthy_leaf.webp")
print("Healthy Leaf Average Color:", healthy_leaf_average)

diseases_leaf_average = extract_color_features("diseased_leaf.jpg")
print("Diseases Leaf Average Color:", diseases_leaf_average)

def extract_shape_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Image is not load!")
        return None

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
    vertex_count = len(approx)

    return [area, perimeter, vertex_count]


healthy_leaf_shape = extract_shape_features("healthy_leaf.webp")
print("Healthy Leaf Shape:", healthy_leaf_shape)

diseases_leaf_shape = extract_shape_features("diseased_leaf.jpg")
print("Diseases Leaf Shape:", diseases_leaf_shape)

