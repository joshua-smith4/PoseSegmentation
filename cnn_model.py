import tensorflow as tf
import keras
from keras.layers import Conv2D, Input, MaxPooling2D, Dropout, Conv2DTranspose, Reshape, Lambda
from keras.models import Model
from load_preproc_data import load_preproc_generator


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
        loss=keras.losses.sparse_categorical_crossentropy,
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
            keras.callbacks.ProgbarLogger(count_mode='steps'),
        ],
    )
