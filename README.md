#  digit/non-digit binary classification in [caffe](https://github.com/BVLC/caffe)
## architecture for this model is similar to [Text-Attentional Convolutional Neural Networks for Scene Text Detection](https://arxiv.org/abs/1510.03283)
## this model is trained to classify digit/non-digit image in ID-CARD scene

## directory structure
+ caffe_ocr/  --dir: a directory contains caffe source code. main change : add file image_data_regression.cpp
+ train_model/  --dir: a directory contains some necessary files to train the model
