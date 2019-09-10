import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from tensorflow import keras


if __name__ == '__main__':

    data = loadmat('./dataset/row_ppg.mat').get('row_ppg')
    label = loadmat('./dataset/labels.mat').get('labels')

    label = label - 1

    X_train = data[:4000, ]
    X_test = data[4000:, ]

    Y_train_orig = label[:4000, ]
    Y_test_orig = label[4000:, ]

    keras.backend.one_hot(Y_train_orig, 3)

    model = keras.Sequential()
    model.add(keras.layers.Reshape((1000, 1)))
    model.add(keras.layers.Conv1D(filters=32, kernel_size=40, activation=tf.nn.relu))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling1D(pool_size=4))
    model.add(keras.layers.Dropout(0.8))
    model.add(keras.layers.Conv1D(filters=32, kernel_size=40, activation=tf.nn.relu))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling1D(pool_size=4))
    model.add(keras.layers.Dropout(0.8))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(3, activation=keras.activations.softmax))

    model.compile(optimizer=keras.optimizers.RMSprop(),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.fit(X_train, Y_train_orig, batch_size=32, epochs=20, validation_data=(X_test, Y_test_orig))



