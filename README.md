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
python cnn_model.py [--load=True]
```
The --load=True option, if included, will load the saved model and continue training from its current state. If the --load=True option is excluded, a new model is created and training starts from the beginning.
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

The image below shows the proposed FCN architecture.

![Model architecture](/model.png?raw=true)

The model accepts depth images with the dimensions batches x 424 x 512. The input is then passed to a reshape layer that converts the input into a tensor with the shape batches x 424 x 512 x 1 because convolutional layers in Keras expect a channels dimension. The output of the reshape step is then sent through several convolutional layers with kernel sizes 3 x 3 and then to a max pooling layer that reduces the dimensionality to 212 x 256 x 32. This is then sent to an upsampling function (Lambda) that performs an image resize bilinear operation to increase dimensionality to the original size with 32 channels. The output of the upsampling function and the original reshape are then combined through addition and sent to a convolution 2D transpose or "deconvolution layer". This increases the channels to 46 and then has a softmax activation on the channels axis. The output is then reshaped to match the needed dimensionality of the custom loss function.

The convolution layers are for feature extraction as they are generally used. The upsampling lambda is necessary to revert to the original image size and the "deconvolution" layer is to maintain locality connectivity during upsampling. The addition layer is meant to help mitigate the resolution loss during upsampling by introducing the original input back into the feature space. This technique is inspired by ResNet.

The output of the network is a tensor with dimensions batches x 217088 x 46. The pixel dimensions have been flattened into vectors of 46 elements where each value represents the probability of that pixel pertaining to the associated class.

### Training (Loss and Evaluation Metrics)

The most difficult part of this project was the construction of a custom loss function that takes into account the large amount of background in each image and weights the foreground data sufficiently to train well. As far as I could find, neither Keras nor Tensorflow have implementations of loss functions for the pixel-wise segmentation problem.

The first iteration, I started with a model that would output a tensor with shape batches x 424 x 512 where each pixel value was some floating point numbers around the range [0,45]. I then performed operations that would round the output and and clip it to the previously mentioned range so that every pixel was assigned a value. I then subtracted this output from the label data which was of the same type, squared the result (element-wise), and summed each element. This was the essence of the first loss function. The idea is that as the squared error decreased, the output and labels would become more and more alike. This loss function proved to be very ineffective in training. I believe that it was due to the rounding functions and the instability they introduced in the gradient when calculated to update the weights. Also, loss functions tend to work well with class probabilities through use of the softmax function but are somewhat numerically unstable when dealing with class labels directly.

This caused me to rework the problem a bit. The current model was born where the output and labels are tensors of size batches x 217,088 x 46 and the softmax activation function is applied on each pixel of the output across the 46 channels to provide true class probabilities. As is common in the classification problem, I chose to use the categorical cross entropy loss function. About halfway through the training I decided to test out the network. I realized that the network was learning the background of the images really well but nothing in the foreground. The background constitutes about 75-85% of each image and would cause the loss function to be very low if the network produced all zero (background) values. The accuracy was very high and the cost function very well minimized but the network was not learning the foreground.

Understanding that the foreground was paramount to the background and that other techniques could be used to eliminate background data after the network produced segmentation information, I decided to implement a custom loss function described in cnn_model.py that ignores background during training. After many tries to get it working right I finally came up with the current format. This loss function finally showed progress where the others hadn't.

A corresponding accuracy metric was developed to ignore background when calculating accuracy as well.

Training took a long time (about 4 hours per epoch). The evaluation metrics and results are descibed below.

### Shortcomings

As expressed [here](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) and in many other texts, FCNs generally provide relatively low resolution segmentation results due to the dimension reduction of convolutional and max pooling layers and the subsequent upsampling. The implemented algorithm suffers from the same weakness. It has been shown in the above text and with ResNet that connecting outputs from much earlier layers to later ones can help by reintroducing the original resolution into the process. I implemented a similar change where the original input is added to the feature space produced by the convolutional and upsampling layers then fed to the last layer for classification.

The network takes a long time to train. The reason it is so shallow is because any deeper or larger network would not fit on the GPU I was using to train and would through error causing the training to stop. Even with the reduction in size, the network would take 4 hours per epoch to train. Training was never able to be fully finished and so the resulting accuracy is very low. The network, as shown below, learned to separate the foreground from the background well but did not learn to perform segmentation of body parts very well.

### External Packages
* [Numpy](https://www.numpy.org/) is a very common matrix/array manipulation and linear algebra package.
* [Tensorflow](https://www.tensorflow.org/) is a graph computation engine produced by google that is very common in machine learning and neural network research.
* [Keras](https://keras.io/) is a wrapper around Tensorflow that facilitates the creation, training, and evaluation of neural networks.

## Changes and Improvements

As mentioned above, the main contributions were:
1. Custom loss function for FCN segmentation that focuses on foreground information
2. Implementation of the technique of carrying input tensors through, unchanged, to some later processes to improve resolution on the output.
3. Use of an FCN over a decision-tree based technique, giving the ability to classify the entire input through a single fully connected network rather than a sliding-window approach.

## Special structures or tricks

As mentioned, the loss function is a special feature of this project. It was developed over many iterations and required the largest amount of time.

Also, I had never worked with a dataset that was too large to fit in memory before. I spent some time developing a generator and the accompanying interface to Keras/Tensorflow that would incrementally load data from the dataset in batches for either training or testing. This generator is custom to the project.

## Difficulties during development

1. The loss function was very difficult to develop. There is not a lot of clear documentation on how to implement a loss function for the FCN segmentation problem and neither is there good documentation on how to implement a custom loss function in Keras. This took a long time to overcome. I am pretty happy with the resulting function but believe that it can be improved further. For instance, rather than ignore the background, I think the contribution of the foreground dimensions to the value of the loss function should be weighted higher than the background (e.g. 80% foreground, 20% background).

2. I have worked with classification networks quite often but had never seen or been introduced to a segmentation framework. I had to research a lot to understand the structures that are necessary and the desired output shape/meaning.

3. The GPU that I am using is not large enough to hold a deeper/larger network than this. This limited the possible architectures that I could experiment with and prolonged training time.

## Experimental Results and Comparison with Shotton et. al.

The final results of the loss function and accuracy are shown below:

Method | Custom Loss Function Value | Accuracy
:--- | :---: | :---:
Proposed FCN | 0.1733 | Custom categorical accuracy: 0.00949 (0.9%)
Shotton et. al. | | Average per-class accuracy: 59%


The above results are produced by running the command
```sh
python cnn_accuracy_test.py
```
as explained in the steps above and after preprocessing the data in the manner described.

Some example outputs of the proposed FCN and those published by Shotton et. al. are shown in the table below.

Label Data | Output of Proposed FCN | Example Shotton et. al. Output
:---: | :---: | :---:
![](/results/gt_pose1.png?raw=true) | ![](/results/nn_pose1.png?raw=true) | ![](/results/sh_pose1.png?raw=true)
![](/results/gt_pose2.png?raw=true) | ![](/results/nn_pose2.png?raw=true) | ![](/results/sh_pose2.png?raw=true)
![](/results/gt_pose3.png?raw=true) | ![](/results/nn_pose3.png?raw=true)

Ideally, the output of the FCN would be identical to the label data on the left.

These images seem to suggest that the FCN was able to correctly learn the segmentation between the background and foreground but suffered at learning the segmentation between different body parts. Shotton et. al. had much better results in distinguishing between body parts.

It should be noted that the two methods compared above used tow different similar datasets. The one used in Shotton et. al. was not included for download. UBC3V was generated in a very similar way but included a larger variety in poses and character models as well as 46 classes where the dataset in the paper had 32.

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
