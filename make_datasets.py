import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from load_preproc_data import load_preproc_generator_windowed, load_preproc_generator


def make_rf_dataset(config, training=True):
    g = load_preproc_generator_windowed(
        config['path_to_ubc3v'],
        config['window_size_x'],
        config['window_size_y'],
        train_split=config['train_split'],
        max_files=config['max_data_files'],
        training_data=training
    )
    def cast_to_tf_gen():
        for x, y in g:
            yield tf.convert_to_tensor(x, dtype=tf.float32, name="input_x"), tf.convert_to_tensor(np.array([y]), dtype=tf.int32, name="input_y")
    return Dataset.from_generator(cast_to_tf_gen, (tf.float32, tf.int32),
                                  (tf.TensorShape([config['window_size_x'] * config['window_size_y']]), tf.TensorShape([1])))


def make_cnn_dataset(config, training=True):
    g = load_preproc_generator(
        config['path_to_ubc3v'],
        train_split=config['train_split'],
        max_files=config['max_data_files'],
        training_data=training,
    )
    def dummy():
        return g
    d = Dataset.from_generator(dummy, (tf.float32, tf.float32))
    d = d.batch(config['batch_size'])
    return d

# def build_estimator(config):
#     params = tensor_forest.ForestHParams(
#         num_classes=46, num_features=config['window_size_y'] * config['window_size_x'],
#         num_trees=config['number_of_trees'], max_nodes=config['max_nodes'])
#     return random_forest.TensorForestEstimator(params, model_dir=config['rf_model_dir'])
