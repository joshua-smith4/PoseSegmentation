import tensorflow as tf

def cnn_model_fn(features, labels, training=True):
    print('shape of features', features.shape)
    input_layer = tf.reshape(features, [-1, 424, 512, 1])
    batch_size_tensor = tf.shape(input_layer)[0]
    conv1w = tf.Variable(
        tf.random.normal([40, 40, 1, 32]),
        dtype=tf.float32,
    )
    conv1 = tf.nn.conv2d(
        input=input_layer,
        filter=conv1w,
        strides=[1, 1, 1, 1],
        padding="SAME",
        name='conv1')
    act1 = tf.nn.relu(conv1, name='act1')

    pool1 = tf.layers.max_pooling2d(
        inputs=act1,
        pool_size=[2, 2],
        strides=2)
    # 212 256 64
    conv2w = tf.Variable(
        tf.random.normal([20, 20, 32, 32]),
        dtype=tf.float32,
        name='conv2w',
    )
    conv2 = tf.nn.conv2d(
        input=pool1,
        filter=conv2w,
        strides=[1, 1, 1, 1],
        padding="SAME",
        name='conv2'
    )
    act2 = tf.nn.relu(conv2, name='act2')

    pool2 = tf.layers.max_pooling2d(
        inputs=act2,
        pool_size=[2, 2],
        strides=2)

    conv3w = tf.Variable(
        tf.random.normal([30, 30, 32, 32]),
        dtype=tf.float32,
        name='conv3w',
    )
    conv3 = tf.nn.conv2d(
        input=pool2,
        filter=conv3w,
        strides=[1, 1, 1, 1],
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
        training=training)

    upsamp1 = tf.tile(
        input=dropout,
        multiples=[1, 2, 2, 1],
        name='upsamp1'
    )

    deconv3_out_shape = tf.stack([batch_size_tensor, 106, 128, 32])
    deconv3 = tf.nn.conv2d_transpose(
        value=upsamp1,
        filter=conv3w,
        output_shape=deconv3_out_shape,
        strides=[1, 1, 1, 1],
        name='deconv3')

    upsamp2 = tf.tile(
        input=deconv3,
        multiples=[1, 2, 2, 1],
        name='upsamp2'
    )

    deconv2_out_shape = tf.stack([batch_size_tensor, 212, 256, 32])
    deconv2 = tf.nn.conv2d_transpose(
        value=upsamp2,
        filter=conv2w,
        output_shape=deconv2_out_shape,
        strides=[1, 1, 1, 1],
        name='deconv2')

    upsamp3 = tf.tile(
        input=deconv2,
        multiples=[1, 2, 2, 1],
        name='upsamp3'
    )


    logits_out_shape = tf.stack([batch_size_tensor, 424, 512, 1])
    logits = tf.nn.conv2d_transpose(
        value=upsamp3,
        filter=conv1w,
        output_shape=logits_out_shape,
        strides=[1, 1, 1, 1],
        name='logits'
    )

    min_logits = tf.reduce_min(logits)
    max_logits = tf.reduce_max(logits)
    logits_scaled = (logits - min_logits) / (max_logits - min_logits) * 45

    classes = tf.round(logits_scaled)
    loss = tf.reduce_sum(
        tf.square(labels - tf.reshape(logits_scaled, [-1, 424, 512])))
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss=loss)
    accuracy = tf.reduce_sum(
        tf.cast(tf.equal(tf.reshape(classes, [-1, 424, 512]), labels), tf.float32)) / (424 * 512)
    return {
        'probability': logits_scaled,
        'classes': classes,
        'loss': loss,
        'accuracy': accuracy,
        'train_op': train_op
    }


if __name__ == '__main__':
    from load_preproc_data import load_preproc_generator
    import json
    import numpy as np

    with open('configuration.json') as f:
        config = json.load(f)


    gen_test = load_preproc_generator(
        config['path_to_ubc3v'],
        config['train_split'],
        config['max_data_files'],
        training_data=False
    )

    x_test, y_test = zip(*[(x_test_i,y_test_i) for x_test_i,y_test_i in gen_test])
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    print('loaded test data: {} {}'.format(x_test.shape, y_test.shape))

    x = tf.placeholder(tf.float32, [None, 424, 512])
    y = tf.placeholder(tf.float32, [None, 424, 512])

    model = cnn_model_fn(x, y, training=True)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    acc_max = 0.0
    with tf.Session() as sess:
        init.run()
        for i in range(config['num_epochs']):
            print('Current Epoch: {}'.format(i))
            # training loop
            gen_train = load_preproc_generator(
                config['path_to_ubc3v'],
                config['train_split'],
                config['max_data_files'],
                training_data=True
            )
            count = 0
            has_training_data = True
            while has_training_data:
                #print('Training Loop Notifier')
                x_batch = []
                y_batch = []
                for j in range(config['batch_size']):
                    try:
                        tmpx, tmpy = next(gen_train)
                        x_batch += [tmpx]
                        y_batch += [tmpy]
                    except StopIteration:
                        has_training_data = False
                        break
                x_batch = np.array(x_batch)
                y_batch = np.array(y_batch)
                count += 1
                print('running batch {}: shape {}'.format(count, x_batch.shape[0]), end='\r', flush=True)
                if x_batch.shape[0] > 0:
                    sess.run(model['train_op'], feed_dict={
                        x: x_batch, y: y_batch})
            # eval test loop
            acc_avg = sess.run(model['accuracy'],
                feed_dict={x: x_test, y: y_test})
            print('Accuracy on test data: {}'.format(acc_avg))
            if acc_avg > acc_max:
                acc_max = acc_avg
                print('Saving epoch {}'.format(i))
                saver.save('sess', config['cnn_model_fp'])
