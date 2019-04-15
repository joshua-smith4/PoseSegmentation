import tensorflow as tf


def cnn_model_fn(features, labels, mode):
    print(features)
    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[40, 40],
        padding="same",
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[20, 20],
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2)

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[30, 30],
        padding="same",
        activation=tf.nn.relu)

    pool3 = tf.layers.max_pooling2d(
        inputs=conv3,
        pool_size=[2, 2],
        strides=2)

    dropout = tf.layers.dropout(
        inputs=pool3, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
