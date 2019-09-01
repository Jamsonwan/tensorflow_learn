import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


def multi_hot_sequences(squences, dimension):
    results = np.zeros((len(squences), dimension))

    for i, word_indices in enumerate(squences):
        results[i, word_indices] = 1.0

    return results


def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label = name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title() + ' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])
    plt.show()


def l2_model(train_datas, train_label, test_datas, test_label, num_data):
    l2_model = keras.models.Sequential([
        keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu,
                           input_shape=(num_data,)),
        keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu,),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    l2_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy', 'binary_crossentropy'])
    l2_model_history = l2_model.fit(train_datas, train_label, epochs=20, batch_size=512,
                                    validation_data=(test_datas, test_label), verbose=2)

    return l2_model_history


def dropout_model(x, y, x_test, y_test, num_data):

    dpt_model = keras.models.Sequential([
        keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(num_data,)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    dpt_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_crossentropy'])

    dpt_history = dpt_model.fit(x, y, epochs=20, batch_size=512,
                  validation_data=(x_test, y_test), verbose=2)

    return dpt_history


if __name__ == '__main__':
    print(tf.__version__)

    NUM_WORDS = 10000
    (train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

    train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
    test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

    # plt.plot(train_data[0])
    # plt.show()

    # l2_history = l2_model(train_data, train_labels, test_data, test_labels, NUM_WORDS)

    baseline_model = keras.Sequential([
        keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    baseline_model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy', 'binary_crossentropy'])
    baseline_model.summary()

    baseline_history = baseline_model.fit(train_data, train_labels, epochs=20,
                                          batch_size=512, validation_data=(test_data, test_labels),
                                          verbose=2)
    dpt_history = dropout_model(train_data, train_labels, test_data, test_labels, NUM_WORDS)
    plot_history([('baseline', baseline_history),
                  ('dropout', dpt_history)])

    '''
    smaller_model = keras.Sequential([
        keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
        keras.layers.Dense(4, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    smaller_model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy', 'binary_crossentropy'])
    smaller_model.summary()

    smaller_history = smaller_model.fit(train_data, train_labels, epochs=20,
                                        batch_size=512, validation_data=(test_data, test_labels),
                                        verbose=2)

    bigger_model = keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    bigger_model.compile(optimizer='adam',
                         loss='binary_crossentropy',
                         metrics=['accuracy', 'binary_crossentropy'])
    bigger_model.summary()
    bigger_history = bigger_model.fit(train_data, train_labels, epochs=20,
                                      batch_size=512, validation_data=(test_data, train_labels),
                                      verbose=2)

    plot_history([('baseline', baseline_history),
                  ('smaller', smaller_history),
                  ('bigger', bigger_history)])
    '''