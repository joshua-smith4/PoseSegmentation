import os
import numpy as np
from sklearn.utils import shuffle
from keras.utils import to_categorical


def load_preproc_generator(fp, batch_size=50, train_split=0.8, training_data=True):
    if type(fp) == bytes:
        files = [f for f in os.listdir(fp) if f.endswith(b'.npz')]
    else:
        files = [f for f in os.listdir(fp) if f.endswith('.npz')]
    files.sort()
    random_state = 0
    counter = 0
    while True:
        handle = np.load(os.path.join(fp, files[counter % len(files)]))
        counter += 1
        x = handle['x']
        y = handle['y']
        x, y = shuffle(x, y, random_state=random_state)
        random_state += 1
        train_divide = int(x.shape[0] * train_split)
        if training_data:
            for i in range(0, train_divide, batch_size):
                x_out = x[i:i + batch_size].astype(np.float32)
                y_out = to_categorical(y[i:i + batch_size].astype(np.float32).reshape(-1,424*512,1)).reshape(-1,424*512*46)
                yield x_out, y_out
        else:
            for i in range(train_divide, x.shape[0], batch_size):
                yield x[i:i + batch_size].astype(np.float32), y[i:i + batch_size].astype(np.float32)


def load_preproc_generator_windowed(fp, wx, wy, pad="edge", padArgs={}, train_split=0.8, max_files=100, training_data=True):
    g = load_preproc_generator(
        fp, train_split=train_split, max_files=max_files, training_data=training_data)
    assert (wx % 2 != 0 and wy %
            2 != 0), "width and height of sliding window must be odd integers"
    counter = 0
    for x, y in g:
        paddedX = np.pad(x, [(wy // 2,), (wx // 2,)], pad, **padArgs)
        for i in range(x.shape[1]):
            for j in range(x.shape[0]):
                window = paddedX[i:i + wy, j:j + wx]
                if np.all(window == 0) and y[i, j] == 0 and counter < 10:
                    counter += 1
                    continue
                counter = 0
                yield window.flatten(), y[i, j]
