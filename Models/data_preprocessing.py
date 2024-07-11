import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    data = []
    labels = []
    for category in ['Uninfected', 'Parasitized']:
        path = os.path.join(data_dir, category)
        class_num = 0 if category == 'Uninfected' else 1
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                resized_img = cv2.resize(img_array, (64, 64))
                data.append(resized_img)
                labels.append(class_num)
            except Exception as e:
                pass
    return np.array(data), np.array(labels)

def preprocess_data(data, labels):
    data = data / 255.0
    return train_test_split(data, labels, test_size=0.2, random_state=42)

if __name__ == "__main__":
    data_dir = 'C:/Data Practice/Malaria Cell Image Dataset'
    data, labels = load_data(data_dir)
    X_train, X_test, y_train, y_test = preprocess_data(data, labels)
    np.save('Models/X_train.npy', X_train)
    np.save('Models/X_test.npy', X_test)
    np.save('Models/y_train.npy', y_train)
    np.save('Models/y_test.npy', y_test)