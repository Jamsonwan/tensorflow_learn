import tensorflow as tf

import numpy as np
from tensorflow.python.framework import ops


def fully_connected(input_layer, num_outputs):
    weight_shape = tf.squeeze(tf.stack([tf.shape(input_layer), [num_outputs]]))

    weight = tf.random_normal(weight_shape, stddev=0.1)
    bias = tf.random_normal(shape=[num_outputs])

    input_layer_2d = tf.expand_dims(input_layer, 0)
    fully_output = tf.add(tf.matmul(input_layer_2d, weight), bias)

    return tf.squeeze(fully_output)


def conv_layer(train_x, filters, stride):
    input_two_dimension = tf.expand_dims(train_x, 0)
    input_three_dimension = tf.expand_dims(input_two_dimension, 0)
    input_four_dimension = tf.expand_dims(input_three_dimension, 3)

    convolution_output = tf.nn.conv2d(input_four_dimension, filters=filters, strides=[1, 1, stride, 1], padding='VALID')
    conv_output = tf.squeeze(convolution_output)

    return conv_output


def activation(train_x):
    return tf.nn.relu(train_x)


def max_pool(train_x, width, stride):
    input_two_dimension = tf.expand_dims(train_x, 0)
    input_three_dimension = tf.expand_dims(input_two_dimension, 0)
    input_four_dimension = tf.expand_dims(input_three_dimension, 3)

    pool_output = tf.nn.max_pool(input_four_dimension, ksize=[1, 1, width, 1], strides=[1, 1, stride, 1], padding='VALID')

    return tf.squeeze(pool_output)


if __name__ == '__main__':
    ops.reset_default_graph()

    sess = tf.compat.v1.Session()

    data_size = 25
    conv_size = 5
    max_pool_size = 5
    stride_size = 1

    seed = 13
    np.random.seed(seed)
    tf.set_random_seed(seed)

    data_one_dimension = np.random.normal(size=data_size)

    x_input_one_dimension = tf.placeholder(dtype=tf.float32, shape=[data_size])

    filters = tf.Variable(tf.random_normal(shape=[1, conv_size, 1, 1]))
    convolution_output = conv_layer(x_input_one_dimension, filters, stride=stride_size)

    activation_output = activation(convolution_output)

    max_pool_output = max_pool(activation_output, width=max_pool_size, stride=stride_size)

    fc_output = fully_connected(max_pool_output, 5)

    init = tf.global_variables_initializer()

    sess.run(init)

    feed_dict = {x_input_one_dimension: data_one_dimension}

    print('>>>> one dimension data <<<<')

    # Convolution output
    print('Input = array of length %d' % (x_input_one_dimension.shape.as_list()[0]))
    print('Convolution w/ filter, length = %d, stride size = %d, results in an array of length %d:' % (conv_size,
                                                                                                       stride_size, convolution_output.shape.as_list()[0]))
    print(sess.run(convolution_output, feed_dict=feed_dict))

    # Activation output
    print('\nInput = above array of length %d' % (convolution_output.shape.as_list()[0]))
    print('ReLU element wise returns an array of length %d:' % (activation_output.shape.as_list()[0]))
    print(sess.run(activation_output, feed_dict=feed_dict))

    # Max pool output
    print('\nInput = above array of length %d' % (activation_output.shape.as_list()[0]))
    print('MaxPool, window length = %d, stride size = %d, results in the array of length %d' % (max_pool_size,
                                                                                                stride_size, max_pool_output.shape.as_list()[0]))
    print(sess.run(max_pool_output, feed_dict=feed_dict))

    # Fully Connected Output

    print('\nInput = above array of length %d' % (max_pool_output.shape.as_list()[0]))
    print('Fully connected layer on all 4 rows with %d output:' % (fc_output.shape.as_list()[0]))
    print(sess.run(fc_output, feed_dict=feed_dict))
