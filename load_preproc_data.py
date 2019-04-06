import os
import numpy as np
from sklearn.utils import shuffle

def load_preproc_data_fcn(fp, train_split=0.8, max_files=100, training_data=True):
    files = [f for f in os.listdir(fp) if f.endswith('.npz')]
    files.sort()
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    random_state = 0
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
        if training_data:
            x_train.append(x[:train_divide])
            y_train.append(y[:train_divide])
        else:
            x_test.append(x[train_divide:])
            y_test.append(y[train_divide:])
    if training_data:
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
    else:
        x_test = np.concatenate(x_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
    if training_data:
        return x_train, y_train
    return x_test, y_test
    # return x_train, y_train, x_test, y_test


def load_preproc_data_rf(fp, wx, wy, pad="edge"):
    pass

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_preproc_data_fcn("/home/jsmith/ubc3v_preproc")
