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

![Network Architecture](https://github.com/AleksBanbur/EE8204---Real-Time-Multi-Task-Learning/blob/master/Images/Network_architecture.PNG?raw=true)
  
The Encoder network is built from the ResNet architecture and supports ResNet [50, 101, 152]. The ResNet architecutre for the 34 layer network is shown below.
 
![ResNet_34](https://github.com/AleksBanbur/EE8204---Real-Time-Multi-Task-Learning/blob/master/Images/ResNEt_34_Arch.png?raw=true)
 
The ResNet architecture employees residual learning which in short is a skip connection that allows the input to a group of layers to skip and be added back to the output of that layer. This can be visualized as mathematically as F(x) + x where x is the input image and F(x) if the input image after convolution-batch normalization-activation have been perform (possibly also pooling for up/down sampling).

![Skip_Connection](https://github.com/AleksBanbur/EE8204---Real-Time-Multi-Task-Learning/blob/master/Images/Skip_Connection.PNG?raw=true)

The encoder network (ResNet) can be broken down into smaller chunks as seen in the ResNet 34 architecutre. The basics for a 34 layer ResNet are:

1.Input Image
    - 224 x 224 x 3 image (RGB)

2. Convolution Layer 1 (grouping layer)
    - Input: Input image
    - Conv: 7x7 kernel, 64 feature maps, stride 2, padding = 3
    - Batch normalization
    - Max Pooling, stride 2
    - Output: Output_Conv_1
        - size = 56 x 56 x 64 (row x column x feature maps)
    
3. Convolution Layer 2 (grouping layer)
    - Input: Output_Conv_1, skip connection to after 2 convolutional layers
    - Conv: 3x3 kernel, 64 feature maps, stride 1, padding = 1
    - Conv: 3x3 kernel, 64 feature maps, stride 1, padding = 1
    - Output: Output_after_2_convolutions + Output_Conv_1
    - Input: Output_after_2_convolutions + Output_Conv_1, skip connection to after 2 convolutional layers
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Output: Output_after_2_convolutions + Output_Conv_1
    - Input: Output_after_2_convolutions + Output_Conv_1, skip connection to after 2 convolutional layers
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Output: Output_after_2_convolutions + Output_Conv_1
    
4. Convolution Layer 3
    - Input: Output_Conv_1, skip connection to after 2 convolutional layers
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Output: Output_after_2_convolutions + Output_Conv_1
    - Input: Output_after_2_convolutions + Output_Conv_1, skip connection to after 2 convolutional layers
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Output: Output_after_2_convolutions + Output_Conv_1
    - Input: Output_after_2_convolutions + Output_Conv_1, skip connection to after 2 convolutional layers
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Output: Output_after_2_convolutions + Output_Conv_1
    - Input: Output_after_2_convolutions + Output_Conv_1, skip connection to after 2 convolutional layers
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Output: Output_after_2_convolutions + Output_Conv_1
    
5. Convolution Layer 4
    - Input: Output_Conv_1, skip connection to after 2 convolutional layers
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Output: Output_after_2_convolutions + Output_Conv_1
    - Input: Output_after_2_convolutions + Output_Conv_1, skip connection to after 2 convolutional layers
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Output: Output_after_2_convolutions + Output_Conv_1
    - Input: Output_after_2_convolutions + Output_Conv_1, skip connection to after 2 convolutional layers
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Output: Output_after_2_convolutions + Output_Conv_1
    - Input: Output_after_2_convolutions + Output_Conv_1, skip connection to after 2 convolutional layers
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Output: Output_after_2_convolutions + Output_Conv_1
    - Input: Output_after_2_convolutions + Output_Conv_1, skip connection to after 2 convolutional layers
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Output: Output_after_2_convolutions + Output_Conv_1
    - Input: Output_after_2_convolutions + Output_Conv_1, skip connection to after 2 convolutional layers
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Output: Output_after_2_convolutions + Output_Conv_1
    
6. Convolution Layer 5
    - Input: Output_Conv_1, skip connection to after 2 convolutional layers
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Output: Output_after_2_convolutions + Output_Conv_1
    - Input: Output_after_2_convolutions + Output_Conv_1, skip connection to after 2 convolutional layers
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Output: Output_after_2_convolutions + Output_Conv_1
    - Input: Output_after_2_convolutions + Output_Conv_1, skip connection to after 2 convolutional layers
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Conv: 3x3 kernel, 64 feature maps, stride 1
    - Output: Output_after_2_convolutions + Output_Conv_1
    
7. Average Pooling

8. Full Connected Layer

The encoder passes the output directly to the Light Weight RefineNet at the output of the encorder and through chained residual pooling blocks. The Light Weight RefineNet implementation is used as a decoder with an architecture as described in the following paper https://arxiv.org/pdf/1810.03272.pdf where modification are made to the original RefineNet to make it more desirable for real time semantic segmentation. A basic idea of the architecture is shown in the picture below:

![Light Weight RefineNet]()


Insert description about light weight refinenet

Finally, the paper makes use of two task branches at the output of the Light Weight Refine Network. Each branch has the same architecture with a 1x1 depth convolution and a 3x3 convolution. Using Multitask learning each branch is able to perform a signle task such as semantic segmentation and depth estimation.
  

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










