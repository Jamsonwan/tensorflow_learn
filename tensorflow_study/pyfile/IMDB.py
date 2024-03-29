import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


def digit_to_word(data):
    word_index = data.get_word_index()

    word_index = {k: (v+3) for k, v in word_index.items()}
    word_index['<PAD>'] = 0
    word_index['<START>'] = 1
    word_index['<UNK>'] = 2
    word_index['<UNUSED>'] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    return reverse_word_index, word_index


def decode_review(text, reverse_word_index):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


def plot_chart(his):
    acc = his.history['acc']
    val_acc = his.history['val_acc']
    loss = his.history['loss']
    val_loss = his.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()



if __name__ == '__main__':
    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    print('train data shape: ', train_data.shape)
    print('train labels shape: ', train_labels.shape)
    print('test data shape: ', test_data.shape)

    reversed_index, word_index = digit_to_word(imdb)
    # print(decode_review(train_data[0], reversed_index))

    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=word_index['<PAD>'],
                                                            padding='post',
                                                            maxlen=256)
    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=word_index['<PAD>'],
                                                           padding='post',
                                                           maxlen=256)

    vocal_size = 10000
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocal_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.summary()

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # dev set 验证集
    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]

    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=40,
                        batch_size=512,
                        validation_data=(x_val, y_val),
                        verbose=1)

    results = model.evaluate(test_data, test_labels)
    print(results)

    history_dict = history.history
    print(history_dict.keys())

    plot_chart(history)
