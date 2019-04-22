import tensorflow as tf
import keras
from keras.layers import Conv2D, Input, MaxPooling2D, Dropout, Conv2DTranspose, Reshape, Lambda
from keras.models import Model


def cnn_model_fn_keras():
    input_shape = (424,512)
    input_layer = Input(shape=input_shape,dtype='float32',name='input_layer')
    in_reshape = Reshape(input_shape + (1,))(input_layer)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_1')(in_reshape)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_2')(conv1)
    pool1 = MaxPooling2D((2,2),strides=(2,2))(conv2)
    # 212 256
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_3')(pool1)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_4')(conv3)
    pool2 = MaxPooling2D((2,2),strides=(2,2))(conv4)
    # 106 128
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_5')(pool2)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_6')(conv5)
    pool3 = MaxPooling2D((2,2),strides=(2,2))(conv6)
    # 53 64
    up1 = Lambda(lambda x: tf.image.resize_bilinear(x,(106,128)))(pool3)
    deconv1 = Conv2DTranspose(512, (3,3), activation='relu', padding='same', name='deconv1')(up1)
    deconv2 = Conv2DTranspose(512, (3,3), activation='relu', padding='same', name='deconv2')(deconv1)

    up2 = Lambda(lambda x: tf.image.resize_bilinear(x,(212,256)))(deconv2)
    deconv3 = Conv2DTranspose(256, (3,3), activation='relu', padding='same', name='deconv3')(up2)
    deconv4 = Conv2DTranspose(256, (3,3), activation='relu', padding='same', name='deconv4')(deconv3)

    up3 = Lambda(lambda x: tf.image.resize_bilinear(x,(424,512)))(deconv4)
    add_input = keras.layers.add([up3,in_reshape])
    deconv5 = Conv2DTranspose(64, (3,3), activation='relu', padding='same', name='deconv5')(add_input)
    deconv6 = Conv2DTranspose(64, (3,3), activation='relu', padding='same', name='deconv6')(deconv5)

    deconv7 = Conv2DTranspose(1, (3,3), activation='relu', padding='same', name='deconv7')(deconv6)

    model = Model(input_layer,deconv7)
    return model


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


#if __name__ == '__main__':
    #from load_preproc_data import load_preproc_generator
    #import json
    #import numpy as np
#
    #with open('configuration.json') as f:
        #config = json.load(f)
#
    #x = tf.placeholder(tf.float32, [None, 424, 512])
    #y = tf.placeholder(tf.float32, [None, 424, 512])
#
    #model = cnn_model_fn(x, y, training=True)
    #init = tf.global_variables_initializer()
    #saver = tf.train.Saver()
    #acc_max = 0.0
    #with tf.Session() as sess:
        #init.run()
        #for i in range(config['num_epochs']):
            #print('Current Epoch: {}'.format(i))
            ## training loop
            #gen_train = load_preproc_generator(
                #config['path_to_ubc3v'],
                #config['train_split'],
                #config['max_data_files'],
                #training_data=True
            #)
            #count = 0
            #has_training_data = True
            #while has_training_data:
                ##print('Training Loop Notifier')
                #x_batch = []
                #y_batch = []
                #for j in range(config['batch_size']):
                    #try:
                        #tmpx, tmpy = next(gen_train)
                        #x_batch += [tmpx]
                        #y_batch += [tmpy]
                    #except StopIteration:
                        #has_training_data = False
                        #break
                #x_batch = np.array(x_batch)
                #y_batch = np.array(y_batch)
                #count += 1
                #print('running batch {}: shape {}'.format(count, x_batch.shape[0]), end='\r', flush=True)
                #if x_batch.shape[0] > 0:
                    #sess.run(model['train_op'], feed_dict={
                        #x: x_batch, y: y_batch})
            ## eval test loop
            #gen_test = load_preproc_generator(
                #config['path_to_ubc3v'],
                #config['train_split'],
                #config['max_data_files'],
                #training_data=False
            #)
            #acc_total = 0.0
            #count = 0
            #for x_test,y_test in gen_test:
                #count += 1
                #x_test = np.expand_dims(x_test, axis=0)
                #y_test = np.expand_dims(y_test, axis=0)
                #acc_total += sess.run(model['accuracy'],
                    #feed_dict={x: x_test, y: y_test})
            #acc_avg = acc_total / count
            #print('Accuracy on test data: {} = {}/{}'.format(acc_avg, acc_total, count))
            #if acc_avg > acc_max:
                #acc_max = acc_avg
                #print('Saving epoch {}'.format(i))
                #saver.save(sess, config['cnn_model_fp'])
