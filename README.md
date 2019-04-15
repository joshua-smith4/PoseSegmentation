# PoseSegmentation
Final project for CS 7680 - image segmentation using depth imagery of the human pose

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
