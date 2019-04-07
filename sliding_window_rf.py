import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.contrib.tensor_forest.client import eval_metrics
from tensorflow.contrib.learn import Estimator
import json
from tensorflow.data import Dataset
from load_preproc_data import load_preproc_generator_windowed


def make_rf_dataset(config):
    return Dataset.from_generator(load_preproc_generator_windowed, (tf.float16, tf.uint8), (
        tf.TensorShape([config['window_size_y'] * config['window_size_x']]), tf.TensorShape([1])))


def build_estimator(config):
    params = tensor_forest.ForestHParams(
        num_classes=46, num_features=config['window_size_y'] * config['window_size_x'],
        num_trees=config['number_of_trees'], max_nodes=config['max_nodes'])
    return random_forest.TensorForestEstimator(params, model_dir=config['rf_model_dir'])


def train_estimator(config):
    estimator = build_estimator(config)
    dataset = make_rf_dataset(config)
    dataset.batch(config['batch_size'])
    iterator = dataset.make_one_shot_iterator()
    x_train, y_train = iterator.get_next()
    def input_fn():
        return x_train, y_train
    estimator.fit(input_fn=input_fn)
    return estimator

with open('configuration.json') as f:
    config = json.load(f)
trained_estimator = train_estimator(config)
