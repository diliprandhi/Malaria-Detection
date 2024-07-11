import numpy as np
from cnn_model import create_cnn_model

def train_model(X_train, y_train):
    input_shape = (64, 64, 3)
    model = create_cnn_model(input_shape)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    model.save('Models/my_model.keras')

if __name__ == "__main__":
    X_train = np.load('Models/X_train.npy')
    y_train = np.load('Models/y_train.npy')
    train_model(X_train, y_train)