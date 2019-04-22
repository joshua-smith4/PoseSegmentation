import tensorflow as tf
import keras
from keras.layers import Conv2D, Input, MaxPooling2D, Dropout, Conv2DTranspose, Reshape, Lambda
from keras.models import Model
from load_preproc_data import load_preproc_generator

from keras.objectives import *
from keras.metrics import binary_crossentropy
import keras.backend as K
import tensorflow as tf


# Softmax cross-entropy loss function for pascal voc segmentation
# and models which do not perform softmax.
# tensorlow only
def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean


def cnn_model_fn_keras():
    input_shape = (424,512)
    input_layer = Input(shape=input_shape,dtype='float32',name='input_layer')
    print(input_layer.shape)
    in_reshape = Reshape(input_shape + (1,))(input_layer)
    print(in_reshape.shape)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_1')(in_reshape)
    print(conv1.shape)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_2')(conv1)
    print(conv2.shape)
    pool1 = MaxPooling2D((2,2),strides=(2,2))(conv2)
    print(pool1.shape)
    # 212 256
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_3')(pool1)
    print(conv3.shape)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_4')(conv3)
    print(conv4.shape)
    pool2 = MaxPooling2D((2,2),strides=(2,2))(conv4)
    print(pool2.shape)
    # 106 128
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_5')(pool2)
    print(conv5.shape)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_6')(conv5)
    print(conv6.shape)
    pool3 = MaxPooling2D((2,2),strides=(2,2))(conv6)
    print(pool3.shape)
    # 53 64
    up1 = Lambda(lambda x: tf.image.resize_bilinear(x,(106,128)))(pool3)
    print(up1.shape)
    deconv1 = Conv2DTranspose(512, (3,3), activation='relu', padding='same', name='deconv1')(up1)
    print(deconv1.shape)
    deconv2 = Conv2DTranspose(512, (3,3), activation='relu', padding='same', name='deconv2')(deconv1)
    print(deconv2.shape)

    up2 = Lambda(lambda x: tf.image.resize_bilinear(x,(212,256)))(deconv2)
    print(up2.shape)
    deconv3 = Conv2DTranspose(256, (3,3), activation='relu', padding='same', name='deconv3')(up2)
    print(deconv3.shape)
    deconv4 = Conv2DTranspose(256, (3,3), activation='relu', padding='same', name='deconv4')(deconv3)
    print(deconv4.shape)

    up3 = Lambda(lambda x: tf.image.resize_bilinear(x,(424,512)))(deconv4)
    print(up3.shape)
    add_input = keras.layers.add([up3,in_reshape])
    print(add_input.shape)
    deconv5 = Conv2DTranspose(64, (3,3), activation='relu', padding='same', name='deconv5')(add_input)
    print(deconv5.shape)
    deconv6 = Conv2DTranspose(64, (3,3), activation='relu', padding='same', name='deconv6')(deconv5)
    print(deconv6.shape)

    deconv7 = Conv2DTranspose(1, (3,3), activation='relu', padding='same', name='deconv7')(deconv6)
    print(deconv7.shape)
    model = Model(input_layer,deconv7)
    print(model.output_shape)
    return model


if __name__ == '__main__':
    num_data_points = 231231
    import json
    with open('configuration.json') as f:
        config = json.load(f)

    gen_train = load_preproc_generator(
        config['path_to_ubc3v'],
        batch_size=config['batch_size'],
        train_split=config['train_split'],
        training_data=True
    )

    model = cnn_model_fn_keras()
    model.compile(
        loss=softmax_sparse_crossentropy_ignoring_last_label,
        optimizer='sgd',
        metrics=['accuracy'],
    )

    model.fit_generator(
        generator=gen_train,
        steps_per_epoch=(num_data_points*config['train_split'])//config['batch_size'],
        epochs=30,
        verbose=2,
        callbacks=[
            keras.callbacks.ModelCheckpoint(config['cnn_model_fp'],verbose=1,save_best_only=True),
            keras.callbacks.ProgbarLogger(count_mode='samples'),
        ],
    )
