from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


def show_one_picture(image):
    plt.figure()
    plt.imshow(image[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()


def value_data_format(images, names, labels):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(names[labels[i]])
    plt.show()


def plot_image(i, predictions_array, true_label, img, names):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel('{} {:2.0f}% ({})'.format(names[predicted_label],
                                         100*np.max(predictions_array),
                                         names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    this_plot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    this_plot[predicted_label].set_color('red')
    this_plot[true_label].set_color('blue')


def plot_test_result(num_row, num_cols, prediction, test_image, test_label, class_name):
    num_images = num_cols * num_row
    plt.figure(figsize=(2*2*num_cols, 2*num_row))

    for i in range(num_images):
        plt.subplot(num_row, 2*num_cols, 2*i+1)
        plot_image(i, prediction, test_label, test_image, class_name)
        plt.subplot(num_row, 2*num_cols, 2*i+2)
        plot_value_array(i, prediction, test_label)

    plt.show()


if __name__ == '__main__':
    print(tf.__version__)

    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print('train data shape: ', train_images.shape)
    print('train label shape: ', train_labels.shape)
    print('test data shape: ', test_images.shape)
    print('test labels shape: ', test_labels.shape)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # value_data_format(train_images, class_names, train_labels)

    # 设置网络
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # 网络的第一层，将输入转化为28*28=784
        keras.layers.Dense(128, activation=tf.nn.relu),  # 网络的第二层为与第一层全连接，该层有128各神经元
        keras.layers.Dense(10, activation=tf.nn.softmax)  # 网络的输出层，该层有10各神经元
    ])

    # 编译模型（模型配置）
    model.compile(optimizer='adam',  # 优化器
                  loss='sparse_categorical_crossentropy',  # 损失函数
                  metrics=['accuracy'])  # 评价指标

    # 训练模型
    model.fit(train_images, train_labels, epochs=5)

    # 模型预测
    predictions = model.predict(test_images)
    # plot_test_result(5, 3, predictions, test_images, test_labels, class_names)

    # 预测单个图形
    img = test_images[0]
    single_img = np.expand_dims(img, 0)
    prediction_single = model.predict(single_img)
    plot_value_array(0, prediction_single, test_labels)
    plt.xticks(range(10), class_names, rotation=45)
    plt.show()