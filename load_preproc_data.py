import os
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.data import Dataset

def load_preproc_generator(fp, train_split=0.8, max_files=100, training_data=True):
    if type(fp) == bytes:
        files = [f for f in os.listdir(fp) if f.endswith(b'.npz')]
    else:
        files = [f for f in os.listdir(fp) if f.endswith('.npz')]
    files.sort()
    random_state = 0
    counter = 0
    for f in files:
        if counter >= max_files:
            return
        counter += 1
        handle = np.load(os.path.join(fp,f))
        x = handle['x']
        y = handle['y']
        x, y = shuffle(x, y, random_state=random_state)
        random_state+=1
        train_divide = int(x.shape[0]*train_split)
        if training_data:
            for i in range(train_divide):
                yield x[i], y[i]
                print('entered generator')
        else:
            for i in range(train_divide, x.shape[0]):
                yield x[i], y[i]


def load_preproc_generator_windowed(fp, wx, wy, pad="edge", padArgs={}, train_split=0.8, max_files=100, training_data=True):
    g = load_preproc_generator(fp, train_split=train_split, max_files=max_files, training_data=training_data)
    assert (wx % 2 != 0 and wy % 2 != 0), "width and height of sliding window must be odd integers"
    counter = 0
    for x,y in g:
        paddedX = np.pad(x, [(wy//2,), (wx//2,)], pad, **padArgs)
        for i in range(x.shape[1]):
            for j in range(x.shape[0]):
                window = paddedX[i:i+wy,j:j+wx]
                if np.all(window == 0) and y[i,j] == 0 and counter < 10:
                    counter += 1
                    print(i,j, 'skipped')
                    continue
                counter = 0
                yield window, y[i,j]
                print(i,j)

tf.reset_default_graph()
sess = tf.Session()
dataset = Dataset.from_generator(load_preproc_generator, (tf.float16, tf.uint8), (tf.TensorShape([424,512]), tf.TensorShape([424,512])), args=(tf.constant('/home/jsmith/ubc3v_preproc'),))
iterator = dataset.make_one_shot_iterator()
x_train, y_train = iterator.get_next()
