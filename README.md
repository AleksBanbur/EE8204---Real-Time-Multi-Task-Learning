# EE8204---Real-Time-Multi-Task-Learning
## Real-Time Joint Semantic Segmentation and Depth Estimation Using Asymmetric Annotations Implementation

### Summary
This repository provides a python implementation using PyTorch for the following research paper:
Title: Real-Time Joint Semantic Segmentation and Depth Estimation Using Asymmetric Annotations
Link: https://arxiv.org/abs/1809.04766

This page will serve as a guide to explain the paper, but also to walk anyone interested in the paper through the steps needed to implement the network in python.
Currently the implementation us in PyTorch but in the future I would like to convert the implementation to TensorFlow.

### Intro
The focus of this paper is to accomplish semantic segmentation and depth estimation using asymmetrical data sets. An asymmetrical data set is simply a data set that contains labels for one of the tasks but not all of them. In the case of this paper the data sets used are the NYUDv2 indoor and KITTI outdoor images. The data set images may have labelled data referencing the semantic map or the depth information.

In order to accomplish the goal of performing both semantic segmentation and depth estimation the author of the paper V. Nekrasov et al. utilizes the following two techniques:
1. Multi-task Learning - Used to create a network that can accomplish both Semantic Segmentation and Depth Estimation
2. Knowldge Distillation - Used to estimate missing label information in data sets based on expert pre-trained teacher network

Data Sets Used to Train the weights
A previous network was used with the NYUDv2 and KITTI outdoor data sets to pre-train the weights. These weights were then used in this implementation to show how the network can quickly perform semantic segmentation and depth estimation.

Dependencies
* --find-links https://download.pytorch.org/whl/torch_stable.html
* torch===1.6.0
* torchvision==0.7.0
* numpy
* opencv-python
* jupyter
* matplotlib
* Pillow

### Network Architecture

The network architecture found in this paper can be broken down into four major parts:
1. Encoder Network
2. Light-Weight Refine Network
3. Chained Residual Pooling blocks
4. Task specific Convolution
    - Segmentation
    - Depth Estimation
  A visual summary of the network architecture is provided below directly from the paper.
  ![Network Architecture](https://github.com/AleksBanbur/EE8204---Real-Time-Multi-Task-Learning/tree/master/Images/NetworkArchitecture.png)
  
  The Encoder network is built from the ResNet architecture and supports ResNet [50, 101, 152]. The

### Paper Implementation

How to run the code
In order to get this code to run I recommend copying the entire repository to your local drive. Create a folder and then use the python venv function to create a local copy of the python interpreter.

The dependencies for this project will be listed below. This specific implementation was done in windows 7 ultimate 64 bit. Using a pretrained network for real time semantic segmentation and depth estimation. The pre trained network can be found in the weights folder. It is possible to pre-train a netowkr with specific weights using a different architecture but in this implementation I decicded to follow the original authors methodoolgy to try and get results as close as possibly to the original paper.

Exmaple of projecy directory creation in command line:
```
mkdir my_Project
-m venv my_Project\venv
```
Extract this repositor to within the folder you created but do not place the venv folder. Once the requirements.txt folder is in your selected folder run the following command in command prompt to download the required libraries/frameworks:
```
my_Project\venv\Scripts\activate.bat
pip install -r requirements.txt
```
### Conclusions

TBA










