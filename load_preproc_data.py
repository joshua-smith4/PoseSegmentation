import os
import numpy as np
from sklearn.utils import shuffle

def load_preproc_generator(fp, train_split=0.8, max_files=100, training_data=True):
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
        else:
            for i in range(train_divide, x.shape[0]):
                yield x[i], y[i]


def load_preproc_generator_windowed(fp, wx, wy, pad="edge", padArgs={}, train_split=0.8, max_files=100, training_data=True):
    g = load_preproc_generator(fp, train_split=train_split, max_files=max_files, training_data=training_data)
    assert (wx % 2 != 0 and wy % 2 != 0), "width and height of sliding window must be odd integers"
    for x,y in g:
        paddedX = np.pad(x, [(wy//2,), (wx//2,)], pad, **padArgs)
        for i in range(x.shape[1]):
            for j in range(x.shape[0]):
                yield paddedX[i:i+wy,j:j+wx], y[i,j]
                print(i,j)

# if __name__ == '__main__':
    # for x,y in load_preproc_generator("/home/jsmith/ubc3v_preproc", training_data=True):
    #     print(x.shape, y.shape)
