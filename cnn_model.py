import tensorflow as tf
import keras
from keras.layers import Conv2D, Input, MaxPooling2D, Dropout, Conv2DTranspose, Reshape, Lambda
from keras.models import Model
from load_preproc_data import load_preproc_generator
import numpy as np

# create keras model of proposed segmentation architecture FCN


def cnn_model_fn_keras():
    # the input shape of the ubc3v dataset
    input_shape = (424, 512)
    input_layer = Input(shape=input_shape, dtype='float32', name='input_layer')
    # keras expects data to be in the format (input_shape,) + (channels,)
    in_reshape = Reshape(input_shape + (1,))(input_layer)
    # convolution layers to extract features
    conv1 = Conv2D(32, (3, 3), activation='relu',
                   padding='same', name='conv2d_1')(in_reshape)
    conv2 = Conv2D(32, (3, 3), activation='relu',
                   padding='same', name='conv2d_2')(conv1)
    # max pooling for dimension reduction
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)
    # up sampling to return to input dimension size
    up1 = Lambda(lambda x: tf.image.resize_bilinear(x, (424, 512)))(pool1)
    # add input to mitigate loss of resolution through upsampling
    added = keras.layers.add([up1, in_reshape])
    # output is a tensor 424 x 512 x 46 (46 class probabilities per pixel)
    logits = Conv2DTranspose(
        46, (3, 3), activation='softmax', padding='same', name='deconv1')(added)
    # flatten output for loss function
    flat_logits = Reshape((424 * 512, 46))(logits)
    # return keras model
    model = Model(input_layer, flat_logits)
    return model

# function used to convert network output to class labels


def model_predict(model, x):
    # get predictions
    y = model.predict(x)
    # reshape to the size of the input
    y = y.reshape(-1, 424, 512, 46)
    # select the highest probability of the 46 classes
    y = np.argmax(y, axis=3)
    return y

# loss function that ignores background class (0) to weight training


def categorical_crossentropy_ignore_first_2d(y_true, y_pred):
    # remove first channel and pass to normal categorical crossentropy
    return keras.losses.categorical_crossentropy(y_true[:, :, 1:], y_pred[:, :, 1:])

# accuracy calculation that ignores background class (0)


def categorical_accuracy_ignore_first_2d(y_true, y_pred):
    # remove first channel and pass to normal categorical crossentropy
    return keras.metrics.categorical_accuracy(y_true[:, :, 1:], y_pred[:, :, 1:])


# call python cnn_model.py to either continue or create and train above model
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # boolean argument saying whether to load model or train new one
    parser.add_argument('--load', type=bool, default=False)
    args = parser.parse_args()

    # because generator is being used, total length of the data is unknown
    # at runtime
    num_data_points = 231231

    # load configuration from configuration.json
    import json
    with open('configuration.json') as f:
        config = json.load(f)

    # generator returning batch sized, shuffled, portions of training/testing data
    gen_train = load_preproc_generator(
        config['path_to_ubc3v'],
        batch_size=config['batch_size'],
        train_split=config['train_split'],
        training_data=True
    )

    # either load existing or create new model for training
    if args.load:
        print('loading model for continued training')
        model = keras.models.load_model(
            config['cnn_model_fp'],
            custom_objects={
                'tf': tf,
                'categorical_accuracy_ignore_first_2d': categorical_accuracy_ignore_first_2d,
                'categorical_crossentropy_ignore_first_2d': categorical_crossentropy_ignore_first_2d,
            }
        )
    else:
        print('building model from the start')
        model = cnn_model_fn_keras()
        model.compile(
            loss=categorical_crossentropy_ignore_first_2d,
            optimizer='sgd',
            metrics=[categorical_accuracy_ignore_first_2d],
        )
    # train model
    model.fit_generator(
        generator=gen_train,
        steps_per_epoch=(num_data_points *
                         config['train_split']) // config['batch_size'],
        epochs=config['num_epochs'],
        verbose=1,
        callbacks=[
            # saves model every epoch
            keras.callbacks.ModelCheckpoint(config['cnn_model_fp'], verbose=1),
            # keras.callbacks.ProgbarLogger(count_mode='steps'),
        ],
    )
