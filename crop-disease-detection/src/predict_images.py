import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

def extract_color_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Image is not load!")
        return None

    avg_color_per_row = np.average(image, axis = 0) #her bir sütun ortları
    avg_color = np.average(avg_color_per_row, axis = 0) # ortların otalaması

    return avg_color

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

def load_dataset(base_path):
    X = []
    y = []

    for label in ["healthy", "diseased"]:
        folder = os.path.join(base_path,label)
        for filename in os.listdir(folder):
            image_path = os.path.join(folder, filename)

            color_features = extract_color_features(image_path)
            shape_features = extract_shape_features(image_path)

            if color_features is not None and shape_features is not None:
                combined_features = np.concatenate([color_features, shape_features])
                X.append(combined_features)
                y.append(0 if label == "healthy" else 1)

    return np.array(X), np.array(y)

X, y = load_dataset("dataset")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


svm_model = SVC()
svm_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier(n_neighbors = 3)
knn_model.fit(X_train, y_train)

def predict_image(path_image, model):
    color_features = extract_color_features(path_image)
    shape_features = extract_shape_features(path_image)

    if color_features is None or shape_features is None:
        print("Could not extract feature from image!")
        return None

    combined_features = np.concatenate([color_features, shape_features]).reshape(1,-1)

    prediction = model.predict(combined_features)

    if prediction[0] == 0:
        print("Prediction: Healthy Leaf!")
    else:
        print("Prediction: Diseased Leaf!")

path_image = "healthy_leaf.webp"

print("SVM Model Prediction for healthy leaf:")
predict_image(path_image, svm_model)

print("\nKNN Model Prediction for healthy leaf:")
predict_image(path_image, knn_model)


path_image2 = "diseased_leaf.jpg"

print("\nSVM Model Prediction for diseased leaf:")
predict_image(path_image2, svm_model)

print("\nKNN Model Prediction for diseased leaf:")
predict_image(path_image2, knn_model)







