import os
import numpy as np
from sklearn.utils import shuffle

def load_preproc_data_fcn(fp, train_split=0.8, max_files=100):
    files = [f for f in os.listdir(fp) if f.endswith('.npz')]
    files.sort()
    random_state = 0
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    counter = 0
    for f in files:
        print(counter)
        if counter >= max_files:
            break
        counter += 1
        handle = np.load(os.path.join(fp,f))
        x = handle['x']
        y = handle['y']
        x, y = shuffle(x, y, random_state=random_state)
        random_state+=1
        train_divide = int(x.shape[0]*train_split)
        x_train.append(x[:train_divide])
        x_test.append(x[train_divide:])
        y_train.append(y[:train_divide])
        y_test.append(y[train_divide:])
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    return x_train, y_train, x_test, y_test


def load_preproc_data_rf(fp, wx, wy, pad="edge"):
    pass
