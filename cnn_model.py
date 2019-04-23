import tensorflow as tf
import keras
from keras.layers import Conv2D, Input, MaxPooling2D, Dropout, Conv2DTranspose, Reshape, Lambda
from keras.models import Model
from load_preproc_data import load_preproc_generator


def cnn_model_fn_keras():
    input_shape = (424,512)
    input_layer = Input(shape=input_shape,dtype='float32',name='input_layer')
    in_reshape = Reshape(input_shape + (1,))(input_layer)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_1')(in_reshape)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_2')(conv1)
    pool1 = MaxPooling2D((2,2),strides=(2,2))(conv2)
    # 212 256
    up1 = Lambda(lambda x: tf.image.resize_bilinear(x,(424,512)))(pool1)
    added = keras.layers.add([up1,in_reshape])
    logits = Conv2DTranspose(46, (3,3), activation='softmax', padding='same', name='deconv1')(added)
    flat_logits = Reshape((424*512,46))(logits)
    model = Model(input_layer,flat_logits)
    return model

def model_predict(model, x):
    y = model.predict(x)
    y = np.argmax(y, axis=3)
    return y

def categorical_crossentropy_ignore_first_2d(y_true, y_pred):
    return keras.losses.categorical_crossentropy(y_true[:,:,1:],y_pred[:,:,1:])

def categorical_accuracy_ignore_first_2d(y_true, y_pred):
    return keras.metrics.categorical_accuracy(y_true[:,:,1:],y_pred[:,:,1:])
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load',type=bool,default=False)
    args = parser.parse_args()

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

    if args.load:
        print('loading model for continued training')
        model = keras.models.load_model(config['cnn_model_fp'],custom_objects={'tf':tf})
    else:
        print('building model from the start')
        model = cnn_model_fn_keras()
        model.compile(
            loss=categorical_crossentropy_ignore_first_2d,
            optimizer='sgd',
            metrics=[categorical_accuracy_ignore_first_2d],
        )

    model.fit_generator(
        generator=gen_train,
       # steps_per_epoch=20,
        steps_per_epoch=(num_data_points*config['train_split'])//config['batch_size'],
        epochs=30,
        verbose=1,
        callbacks=[
            keras.callbacks.ModelCheckpoint(config['cnn_model_fp'],verbose=1),
            # keras.callbacks.ProgbarLogger(count_mode='steps'),
        ],
    )
