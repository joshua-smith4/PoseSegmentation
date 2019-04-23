import tensorflow as tf
import keras
from keras.layers import Conv2D, Input, MaxPooling2D, Dropout, Conv2DTranspose, Reshape, Lambda
from keras.models import Model
from load_preproc_data import load_preproc_generator


def cnn_model_fn_keras():
    input_shape = (424,512)
    input_layer = Input(shape=input_shape,dtype='float32',name='input_layer')
    print(input_layer.shape)
    in_reshape = Reshape(input_shape + (1,))(input_layer)
    print(in_reshape.shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_1')(in_reshape)
    print(conv1.shape)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_2')(conv1)
    print(conv2.shape)
    pool1 = MaxPooling2D((2,2),strides=(2,2))(conv2)
    print(pool1.shape)
    # 212 256
    up1 = Lambda(lambda x: tf.image.resize_bilinear(x,(424,512)))(pool1)

    print(up1.shape)
    logits = Conv2DTranspose(46, (3,3), padding='same', name='deconv1')(up1)
    print(logits.shape)
    flat_logits = Reshape((424*512,46))(logits)
    print(flat_logits.shape)
    # probs = keras.activations.softmax(logits, axis=3)
    # print(probs.shape)
    model = Model(input_layer,flat_logits)
    print(model.output_shape)
    return model

def model_predict(model, x):
    y = model.predict(x)
    y = y.reshape(-1, 424,512,46)
    y = np.argmax(y, axis=3)
    return y

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
            loss=keras.losses.categorical_crossentropy,
            optimizer='sgd',
            metrics=['categorical_accuracy'],
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
