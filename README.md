# PoseSegmentation
Final project for CS 7680 - image segmentation using depth imagery of the human pose

## Work to date - 4/7/2019
1. Added preprocessing code to convert UBC3V to compatible format with Tensorflow
2. Created generators to incrementally load chunks of dataset. This was done to reduce the memory requirements of the program.
3. Separated generators into full and windowed versions. Full retrieves full depth images and full segmented images. Windowed splits each image into ~217,088 windows for each pixel and returns a single class value per window.
4. Research into implementing random forests in Tensorflow. This has taken a long time. There is not a lot of documentation on this as it is part of the contrib module and not the core of Tensorflow.
5. Began implementing random forest and developed dataset feeder and training function for it.

