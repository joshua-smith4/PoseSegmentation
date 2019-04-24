# Human Pose Segmentation with FCN
Final project for CS 7680 - image segmentation using depth imagery of the human pose

## Steps to run code
1. Download UBC3V dataset from [here](https://github.com/ashafaei/ubc3v).
2. Ensure that the packages and version numbers in requirements.txt match the current python environment. The code is written to be compatible with python 3.6.
3. Run
```sh
python preproc_ubc3v.py -s=[train,test,valid] -ss=[subsection] -p=[easy,inter,hard] -b=/path/to/ubc3v/base/folder
```
This command should be run multiple times on many subsections and train/test data in order to preprocess data into the numpy format. All the generated .npz files should be placed in a directory we will call ubc3v_preproc.
4. Edit configuration.json to refect the current environment. (e.g. path_to_ubc3v should be /path/to/ubc3v_preproc, cnn_model_fp should be path to saved CNN model in cnn_model directory).
5. Training is performed by running the following command
```sh
python cnn_model.py [--load]
```
The --load option, if included, will load the saved model and continue training from its current state. If the --load option is excluded, a new model is created and training starts from the beginning.
6. Evaluation is performed by running the following command
```sh
python cnn_accuracy_test.py
```
This loads in the saved model from cnn_model and creates a generator the incrementally loads test data from ubc3v_preproc and evaluates the model using the custom metric categorical_accuracy_ignore_first_2d defined in cnn_model.py.

### File descriptions
File | Description
--- | ---
cnn_model/ubc3v_cnn_model.chkpt | Trained saved Keras model for performing image segmentation on ubc3v_preproc dataset
cnn_accuracy_test.py | Script for evaluating the categorical accuracy of the saved model on testing data
cnn_model.py | File containing the model description and functions to build model, custom loss function, custom accuracy measure, and train model
configuration.json | File containing configuration parameters for the entire project
load_preproc_data.py | File containing function to create generators for incrementally loading and returning training and testing data from ubc3v_preproc.
model.png | Image of model architecture
preproc_ubc3v.py | Functions and script for preprocessing ubc3v dataset into numpy arrays for use in training or testing.
README.md | Project description and report
requirements.txt | Project dependency list

## Time to execute code
The training process is really slow. One should expect training to take about 4 hours per epoch depending on the hardware and configuration. Testing (cnn_accuracy_test.py) takes about 45 minutes to return an accuracy measure.

## System components
### Dataset
The UBC3V dataset is a synthetically generated dataset for human pose estimation. Image/input data is stored as 8-bit grayscale images which can be converted to depth data using the following function:
```
depth_data = orig_data / 255.0 * (8 - 0.5) + 0.5
```
Label data are stored as RGBA images where the alpha channel is either 0 (background) or 255 (foreground) and different color combinations of RGB represent different body regions. UBC3V has 45 different labels for body regions. Those plus the background makes 46 classes.

The preprocessing on UBC3V converts the grayscale images to depth data using the above function and labels to images where each pixel is a value in the range [0,45]. The preprocessing was adapted from the [Matlab API](https://github.com/ashafaei/ubc3v) to be compatible with python and the libraries used in this project.

As the data is loaded into memory to be fed to the FCN for training or testing, the labels are converted to categorical arrays and flattened. Each label is a 2D vector with 424*512 elements in the first dimension (one for each pixel) and 46 elements in the second where only a single element is 1 and the rest are 0s. The 46 elements represent the 46 possible classes.

### The FCN model
![Model architecture](/model.png?raw=true)
### Training (Loss and Evaluation Metrics)
### Shortcomings
### External Packages
* Numpy
* Tensorflow
* Keras

## Changes and Improvements
## Special structures or tricks
## Difficulties during development
## Experimental Results and Comparison with Shotton et. al.
Custom Loss Function: 0.173386267145078
Custom Categorical Accuracy: 0.00949561961465289

Label Data | Output of Proposed FCN | Example Shotton et. al. Output
:---: | :---: | :---:
![](/results/gt_pose1.png?raw=true) | ![](/results/nn_pose1.png?raw=true) | ![](/results/sh_pose1.png?raw=true)
![](/results/gt_pose2.png?raw=true) | ![](/results/nn_pose2.png?raw=true) | ![](/results/sh_pose2.png?raw=true)
![](/results/gt_pose3.png?raw=true) | ![](/results/nn_pose3.png?raw=true)



## Notes
Training time

## Work to date - 4/7/2019
1. Added preprocessing code to convert UBC3V to compatible format with Tensorflow
2. Created generators to incrementally load chunks of dataset. This was done to reduce the memory requirements of the program.
3. Separated generators into full and windowed versions. Full retrieves full depth images and full segmented images. Windowed splits each image into ~217,088 windows for each pixel and returns a single class value per window.
4. Research into implementing random forests in Tensorflow. This has taken a long time. There is not a lot of documentation on this as it is part of the contrib module and not the core of Tensorflow.
5. Began implementing random forest and developed dataset feeder and training function for it.

## Work to date - 4/14/2019
1. A lot of work becoming more familiar with Tensorflow. Specifically the Dataset class that allows for incrementally loading large datasets and feeding it to a network for training/testing.
2. Finalized the model for the CNN that will be used for segmentation. 3 convolutional layers with pooling in between, a dropout layer, and 3 deconv/conv2d_transpose layers.
3. Wrote the training code for the CNN. Training is going to start at 40 epochs with batch size of 35 over 184,984 training images. Accuracy over 46,246 testing images will be used for a termination condition.
4. Having some trouble with the dimensions of the deconv layers. After this error is resolved training will commence.
