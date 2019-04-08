import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.contrib.tensor_forest.client import eval_metrics
from tensorflow.contrib.learn import Estimator
import json
from tensorflow.data import Dataset
from load_preproc_data import load_preproc_generator_windowed
# tf.enable_eager_execution()


def make_rf_dataset(config):
    g = load_preproc_generator_windowed(config['path_to_ubc3v'], config['window_size_x'], config['window_size_y'])
    def cast_to_tf_gen():
        for x,y in g:
            yield tf.cast(x,tf.float32), tf.cast(y,tf.int32)
    return Dataset.from_generator(cast_to_tf_gen, (tf.float32, tf.int32), (
        tf.TensorShape([config['window_size_y'] * config['window_size_x']]), tf.TensorShape([])))


def build_estimator(config):
    params = tensor_forest.ForestHParams(
        num_classes=46, num_features=config['window_size_y'] * config['window_size_x'],
        num_trees=config['number_of_trees'], max_nodes=config['max_nodes'])
    return random_forest.TensorForestEstimator(params, model_dir=config['rf_model_dir'])


def train_estimator(config):
    estimator = build_estimator(config)

    def input_fn():
        dataset = make_rf_dataset(config)
        dataset.batch(config['batch_size'])
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
    estimator.fit(input_fn=input_fn)
    return estimator


with open('configuration.json') as f:
    config = json.load(f)
trained_estimator = train_estimator(config)

# dataset = make_rf_dataset(config)
# dataset.batch(config['batch_size'])
# iterator = dataset.make_one_shot_iterator()
# x, y = iterator.get_next()
# sess = tf.Session()
# x_f, y_f = sess.run((x, y))
# print(x_f.shape, y_f.shape)
