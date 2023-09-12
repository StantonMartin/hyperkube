import numpy as np
import pandas as pd
import os
import sklearn
import sklearn.metrics as met
from sklearn.model_selection import KFold

from tqdm import tqdm

### ML models
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor

### Keras/TF
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def get_model(c=80, l=1, shape=(10, 10)):
    model = Sequential()

    model.add(
        Conv1D(
            filters=c,
            kernel_size=3,
            activation="relu",
            input_shape=shape,
        )
    )
    model.add(MaxPooling1D(pool_size=2))

    for ll in range(l):
        model.add(Conv1D(filters=c, kernel_size=3, activation="relu"))
        model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(c, activation="relu"))
    model.add(Dense(1, activation="linear"))

    # compile and train model
    model.compile(loss="mean_squared_error", optimizer="nadam")

    return model


#
# CNN-based Regression
# Describe phenotypes using provided hyperspectal signatures.
# The model is implemented using 1-dimensional convolutional neural networks.
# Input:
#       train_X - hyperspectral signatures of training samples
#       train_Y - target phenotypes of training samples
#       test_X - hyperspectral signatures of testing samples
#       test_Y - target phenotypes of testing samples
#       nodes_per_layer - number of convolution filter per each layer
#       n_layer - number of convolution layers
# Output: R2 score
#
def regression_cnn(train_X, train_Y, test_X, test_Y, nodes_per_layer=60, n_layers=4):
    model = get_model(
        c=nodes_per_layer, l=n_layers, shape=(train_X.shape[1], train_X.shape[2])
    )

    stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(
        train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2]),
        train_Y,
        epochs=500,
        verbose=1,
        batch_size=10,
        validation_split=0.1,
        callbacks=[stopper],
    )

    y_pred_cnn = model.predict(
        test_X.reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2])
    )

    print(met.r2_score(test_Y, y_pred_cnn))
