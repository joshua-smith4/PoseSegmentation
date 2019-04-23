from keras.models import load_model
import tensorflow as tf
from load_preproc_data import load_preproc_generator
import json

# get configuration
with open('configuration.json') as f:
    config = json.load(f)

# create generator for test data
gen_test = load_preproc_generator(
    config['path_to_ubc3v'],
    batch_size=1,
    train_split=config['train_split'],
    training_data=False
)

num_data_points = 231231
# load saved model from training
model = load_model(config['cnn_model_fp'], custom_objects={'tf':tf})

# evalute model on test data
model.evaluate_generator(
    generator=gen_test,
    steps=num_data_points*(1-config['train_split'])-1,
    verbose=1,
)
