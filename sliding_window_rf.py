import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.contrib.tensor_forest.client import eval_metrics
from tensorflow.contrib.learn.python.learn.estimators import estimator
import json
from tensorflow.data import Dataset
from load_preproc_generator import load_preproc_generator_windowed

with open('configuration.json') as f:
    config = json.load(f)

tf.reset_default_graph()
sess = tf.Session()

dataset = Dataset.from_generator(load_preproc_generator_windowed, (tf.float16, tf.uint8), (
    tf.TensorShape([config['window_size_y'], config['window_size_x']]), tf.TensorShape([1])))


def build_estimator(model_dir):
    params = tensor_forest.ForestHParams(
        num_classes=46, num_features=config['window_size_y']*config['window_size_x'],
        num_trees=config['number_of_trees'], max_nodes=config['max_nodes'])
    graph_builder_class = tensor_forest.RandomForestGraphs
    return estimator.SKCompat(random_forest.TensorForestEstimator(
        params, graph_builder_class=graph_builder_class,
        model_dir=model_dir))
