import tensorflow as tf


def cnn_model_fn(features, labels, mode):
    print('shape of features',features.shape)
    input_layer = tf.reshape(features, [-1, 424, 512, 1])

    conv1w = tf.Variable(
        tf.random.normal([40,40,1,64]),
        dtype=tf.float32,
    )
    conv1 = tf.nn.conv2d(
        input=input_layer,
        filter=conv1w,
        strides=[1,1,1,1],
        padding="SAME",
        name='conv1')
    act1 = tf.nn.relu(conv1, name='act1')

    pool1 = tf.layers.max_pooling2d(
        inputs=act1,
        pool_size=[2, 2],
        strides=2)

    conv2w = tf.Variable(
        tf.random.normal([20,20,64,128]),
        dtype=tf.float32,
        name='conv2w',
    )
    conv2 = tf.nn.conv2d(
        input=pool1,
        filter=conv2w,
        strides=[1,1,1,1],
        padding="SAME",
        name='conv2'
    )
    act2 = tf.nn.relu(conv2, name='act2')

    pool2 = tf.layers.max_pooling2d(
        inputs=act2,
        pool_size=[2, 2],
        strides=2)

    conv3w = tf.Variable(
        tf.random.normal([30,30,128,64]),
        dtype=tf.float32,
        name='conv3w',
    )
    conv3 = tf.nn.conv2d(
        input=pool2,
        filter=conv3w,
        strides=[1,1,1,1],
        padding="SAME",
        name='conv3'
    )
    act3 = tf.nn.relu(conv3, name='act3')

    pool3 = tf.layers.max_pooling2d(
        inputs=act3,
        pool_size=[2, 2],
        strides=2)

    dropout = tf.layers.dropout(
        inputs=pool3,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN)

    deconv3 = tf.nn.conv2d_transpose(
        value=dropout,
        filter=conv3w,
        output_shape=[-1,106,128,128],
        strides=[1,1,1,1],
        name='deconv3')

    deconv2 = tf.nn.conv2d_transpose(
        value=deconv3,
        filter=conv2w,
        output_shape=[-1,212,256,64],
        strides=[1,1,1,1],
        name='deconv2')

    logits = tf.nn.conv2d_transpose(
        value=deconv2,
        filter=conv1w,
        output_shape=[-1,424,512,1],
        strides=[1,1,1,1],
        name='logits')
    min_logits = tf.reduce_min(logits)
    max_logits = tf.reduce_max(logits)
    logits_scaled = (logits - min_logits)/(max_logits - min_logits)*45

    classes = tf.round(logits_scaled)
    loss = tf.reduce_sum(tf.square(labels - logits_scaled))
    return {
        'probability': logits_scaled,
        'classes': classes,
        'loss': loss,
    }
