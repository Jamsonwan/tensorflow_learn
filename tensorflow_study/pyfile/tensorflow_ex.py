import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

DATA_FILE = 'boston_housing.csv'
BATCH_SIZE = 10
NUM_FEATURES = 14


def generate_data(feature_batch, label_batch):
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for _ in range(5): # Generate 5 batches
            features, labels = sess.run([feature_batch, label_batch])
            print(features, 'HI')
        coord.request_stop()
        coord.join(threads)


def data_generator(filename):
    f_queue = tf.train.string_input_producer(filename)
    reader = tf.TextLineReader(skip_header_lines=1) # skips the first line

    _, value = reader.read(f_queue)

    record_defaults = [[0.0] for _ in range(NUM_FEATURES)]

    data = tf.decode_csv(value, record_defaults=record_defaults)
    features = tf.stack(tf.gather_nd(data, [[5], [10], [12]]))
    label = data[-1]

    dequeuemin_after_dequeue = 10 * BATCH_SIZE
    capacity = 20 * BATCH_SIZE

    feature_batch, label_batch = tf.train.shuffle_batch([features, label], batch_size=BATCH_SIZE,
                                                        min_after_dequeue=dequeuemin_after_dequeue, capacity=capacity)

    return feature_batch, label_batch


# 固定输入值将权重和偏量结合起来
def append_bias_reshape(features, labels):
    m = features.shape[0]
    n = features.shape[1]
    x = np.reshape(np.c_[np.ones(m), features], [m, n+1])
    y = np.reshape(labels, [m, 1])
    return x, y


# 将数据进行归一化处理
def normalize(X):
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / std

    return X


def tutor():
    coefficients = np.array([[1], [-20], [25]])

    w = tf.Variable([0], dtype=tf.float32)
    x = tf.placeholder(tf.float32, [3, 1])

    cost = x[0][0] * w ** 2 + x[1][0] * w + x[2][0]

    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        print(session.run(w))

        for i in range(1000):
            session.run(train, feed_dict={x: coefficients})

        print(session.run(w))


def simple_LR():
    boston = tf.contrib.learn.datasets.load_dataset('boston')
    X_train, Y_train = boston.data[:, 5], boston.target
    # X_train = normalize(X_train)
    n_samples = len(X_train)

    X = tf.placeholder(tf.float32, name='X')
    Y = tf.placeholder(tf.float32, name='Y')

    b = tf.Variable(0.0)
    w = tf.Variable(0.0)

    Y_hat = X * w + b
    loss = tf.square(Y - Y_hat, name='loss')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    init_op = tf.global_variables_initializer()
    total = []

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        writer = tf.summary.FileWriter('graphs', sess.graph)

        for i in range(100):
            total_loss = 0
            for x, y in zip(X_train, Y_train):
                _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
                total_loss += l
            total.append(total_loss / n_samples)
            print('Epoch {0}: Loss {1}'.format(i, total_loss / n_samples))
        writer.close()
        b_value, w_value = sess.run([b, w])

    Y_pred = X_train * w_value + b_value
    print('Done')

    plt.plot(X_train, Y_train, 'bo', label='Real Data')
    plt.plot(X_train, Y_pred, 'r', label='Predicted Data')
    plt.legend()
    plt.show()
    plt.plot(total)
    plt.show()


def multip_LR():

    boston = tf.contrib.learn.datasets.load_dataset('boston')
    X_train, Y_train = boston.data, boston.target

    print(X_train.shape)
    X_train = normalize(X_train)
    X_train, Y_train = append_bias_reshape(X_train, Y_train)

    m = len(X_train)  # Number of training examples
    n = 13 + 1  # Number of features + bias

    X = tf.placeholder(tf.float32, name='X', shape=[m, n])
    Y = tf.placeholder(tf.float32, name='Y')

    w = tf.Variable(tf.random_normal([n, 1]))

    Y_hat = tf.matmul(X, w)

    loss = tf.reduce_mean(tf.square(Y - Y_hat, name='loss'))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    init_op = tf.global_variables_initializer()
    total = []

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)  # Initialize variable
        writer = tf.summary.FileWriter('graphs', sess.graph)
        for i in range(100):
            _, l = sess.run([optimizer, loss], feed_dict={X: X_train, Y: Y_train})
            total.append(l)
            print('Epoch {0}: Loss {1}'.format(i, l))
        writer.close()
        w_value = sess.run(w)

    plt.plot(total)
    plt.show()

    N = 500
    X_new = X_train[N, :]
    # Y_new = Y_train[N, :]
    # X_new, Y_new = append_bias_reshape(X_new, Y_new)
    Y_pred = np.matmul(X_new, w_value).round(1)
    print('Predicted value: ${0} Actual value: / ${1}'.format(Y_pred[0]*1000, Y_train[N]*1000, '\nDone'))


if __name__ == '__main__':

    # feature_batch, label_batch = data_generator([DATA_FILE])
    # generate_data(feature_batch, label_batch)

    multip_LR()
