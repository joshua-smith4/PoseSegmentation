import os
import numpy as np
from sklearn.utils import shuffle
from keras.utils import to_categorical

# this generator incrementally loads training/testing data from the preprocessed dataset
# fp: file path to directory containing preprocessed data
# train_split: percent of data used to train
# training_data: generate training or testing data
def load_preproc_generator(fp, batch_size=50, train_split=0.8, training_data=True):
    if type(fp) == bytes:
        files = [f for f in os.listdir(fp) if f.endswith(b'.npz')]
    else:
        files = [f for f in os.listdir(fp) if f.endswith('.npz')]
    files.sort()
    random_state = 0
    counter = 0
    # the training function "fit" requires the generator to loop forever and
    # continually provide data
    while True:
        # load preprocessed data from file
        handle = np.load(os.path.join(fp, files[counter % len(files)]))
        counter += 1
        # extract image data
        x = handle['x']
        # extract label data
        y = handle['y']
        # shuffle data uniformly
        x, y = shuffle(x, y, random_state=random_state)
        random_state += 1
        # get train_divide index
        train_divide = int(x.shape[0] * train_split)
        # switch on whether providing training or testing data
        if training_data:
            # loop in batches
            for i in range(0, train_divide, batch_size):
                # extract x
                x_out = x[i:i + batch_size].astype(np.float32)
                # change y to flattened categorical data for loss function
                y_out = to_categorical(y[i:i + batch_size].astype(np.float32).reshape(-1,424*512,1),num_classes=46)
                yield x_out, y_out
        else:
            # same as above except with testing data
            for i in range(train_divide, x.shape[0], batch_size):
                x_out = x[i:i + batch_size].astype(np.float32)
                y_out = to_categorical(y[i:i + batch_size].astype(np.float32).reshape(-1,424*512,1),num_classes=46)
                print(y_out.shape)
                yield x_out, y_out
