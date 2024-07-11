import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

def evaluate_model(X_test, y_test):
    model = tf.keras.models.load_model('Models/my_model.keras')
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    X_test = np.load('Models/X_test.npy')
    y_test = np.load('Models/y_test.npy')
    evaluate_model(X_test, y_test)