# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import torch
import torch.nn as nn
import math

#Creating a 3x3 convolution definition
def conv3x3(in_channel, out_channel, stride = 1, bias = False, dilation = 1, groups = 1):
    "Creating a method for 2D convolution of a 3x3 kernel"
    #Inputs to the definition are:
    #Input channel size - Number of channels in the input image
    #Output channel size - Number of channels produced by the convolution
    #Stride = 1, Stride of the convolution
    #bias = False, Setting the bias to be learnable or not
    #Dilation = 1, Spacing between kernel elements during convolution
    #Groups = 1, Number of blocked connections from input channels to output channels
    return nn.Conv2d(in_channel, out_channel, kernel_size = 3, stride = stride, padding = dilation, dilation = dilation, bias = bias, groups = groups)

#Creating a 1x1 convolution definition
def conv1x1(in_channel, out_channel, stride = 1, bias = False, groups = 1):
    "Creating a method for 2D convolution of a 1x1 kernel"
    #Inputs to the definition are:
    #Input channel size - Number of channels in the input image
    #Output channel size - Number of channels produced by the convolution
    #Stride = 1, Stride of the convolution
    #Groups = 1, Number of blocked connections from input channels to output channels
    return nn.Conv2d(in_channel, out_channel, kernel_size = 1, stride = stride, padding = 0, bias = bias, groups = groups)

#Creating a batch normalization definition
def batch_norm(num_features):
    "Creating a method for 2D batch normalization"
    #Inputs to batchnorm2d:
    #Number of features - An expected input of size C
    #Eps - Denominator value in batch norm equation added for stability
    #Momentum - Value used to calculate running mean and running var computation
    #Affine - Boolean value that when set to true, the module has learnable affine parameters
    return nn.BatchNorm2d(num_features, eps = 1e-5, momentum = 0.1, affine = True)

#Creating the conv-bn-relu sequence
def con_bn_act(in_channel, out_channel, kernel_size, stride = 1, groups = 1, act = True):
    "Creating a method for the convolution, batch normalization, and activation using ReLU using the PyTorch nn.Sequential function"
    if act:
        return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride = stride, padding = int(kernel_size / 2.), groups = groups, bias = False), batch_norm(out_channel), nn.ReLU6(inplace = True))
    else:
        return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride = stride, padding = int(kernel_size / 2.), groups = groups, bias = False), batch_norm(out_channel))


#Creating the Chained Residual Pooling class
#This class is a child class of the PyTorch Parent nn.Module
class chained_residual_pooling(nn.Module):
    "This is the Chained Residual Pooling class"
    #Constructor method __init__() is used to initialize class variables
    #Input channel size
    #Output channel size
    #Number of stages
    #Groups
    def __init__(self, in_channel, out_channel, n_stages, groups = False):
        #Using the super function we are able to call the __init__() method of the nn.Module parent
        #In this case super(chained_residual_pooling, self).__init__() = super().__init__()
        #This is becuase the first argument of super is the same as the class we are calling from within
        super(chained_residual_pooling, self).__init__()
        for i in range(n_stages):
            setattr(self, "{}_{}".format(i + 1, 'outvar_dimred'), conv1x1(in_channel if (i == 0) else out_channel, out_channel, stride = 1, bias=False, groups = in_channel if groups else 1))
        
        #Initializing class variables for object instantiating using the self parameter
        #Using self will allow the current instance of the class to be linked the object calling the class
        self.stride = 1 #Setting the stride
        self.n_stages = n_stages #Setting the number of stages
        self.maxpool = nn.MaxPool2d(kernel_size = 5, stride = 1, padding = 2) #Defining maxpool as the PyTorch nn.MaxPool2d method
    
    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x

#Creating the Inverted Residual block
#This block was taken directly from the paper in the link with minor adjustments to variable names
class Inverted_Residual_Block(nn.Module):
    """Inverted Residual Block from https://arxiv.org/abs/1801.04381"""
    def __init__(self, in_channel, out_channel, expansion_factor, stride = 1):
        super(Inverted_Residual_Block, self).__init__()
        intermed_channel = in_channel * expansion_factor
        self.residual = (in_channel == out_channel) and (stride == 1)
        self.output = nn.Sequential(con_bn_act(in_channel, intermed_channel, 1),
                                    con_bn_act(intermed_channel, intermed_channel, 3, stride = stride, groups = intermed_channel),
                                    con_bn_act(intermed_channel, out_channel, 1, act = False))

    def forward(self, x):
        residual = x
        out = self.output(x)
        if self.residual:
            return (out + residual)
        else:
            return out
        

#Creating the network Architecture
class network_Arch(nn.Module):
    """"Real Time Semantic Segmenataion and Depth Estimation Neural Network Arch"""
    mobile_Net_Config = [[1, 16, 1, 1],
                         [6, 24, 2, 2],
                         [6, 32, 3, 2],
                         [6, 64, 4, 2],
                         [6, 96, 3, 1],
                         [6, 160, 3, 2],
                         [6, 320, 1, 1],
                         ]
    
    in_channel = 32
    num_layers = len(mobile_Net_Config)
    def __init__(self, num_classes, num_tasks = 2):
        super(network_Arch, self).__init__()
        self.num_tasks = num_tasks
        assert self.num_tasks in [2, 3], "Number of tasks supported is either 2 or 3, got {}".format(self.num_tasks)

        self.layer1 = con_bn_act(3, self.in_channel, kernel_size=3, stride=2)
        c_layer = 2
        for t,c,n,s in (self.mobile_Net_Config):
            layers = []
            for idx in range(n):
                layers.append(Inverted_Residual_Block(self.in_channel, c, expansion_factor = t, stride = s if idx == 0 else 1))
                self.in_channel = c
            setattr(self, 'layer{}'.format(c_layer), nn.Sequential(*layers))
            c_layer += 1
            
        #Creating the Leight-Weight Refine Network Architecture
        self.conv8 = conv1x1(320, 256, bias=False) #in_channel = 320, out_channel = 256
        self.conv7 = conv1x1(160, 256, bias=False) #in_channel = 160, out_channel = 256
        self.conv6 = conv1x1(96, 256, bias=False) #in_channel = 96, out_channel = 256
        self.conv5 = conv1x1(64, 256, bias=False) #in_channel = 64, out_channel = 256
        self.conv4 = conv1x1(32, 256, bias=False) #in_channel = 32, out_channel = 256
        self.conv3 = conv1x1(24, 256, bias=False) #in_channel = 24, out_channel = 256
        self.crp4 = self._make_crp(256, 256, 4, groups=False) #in_channel = 256, out_channel = 256, stages = 4
        self.crp3 = self._make_crp(256, 256, 4, groups=False) #in_channel = 256, out_channel = 256, stages = 4
        self.crp2 = self._make_crp(256, 256, 4, groups=False) #in_channel = 256, out_channel = 256, stages = 4
        self.crp1 = self._make_crp(256, 256, 4, groups=True) #in_channel = 256, out_channel = 256, stages = 4, groups = True

        self.conv_adapt4 = conv1x1(256, 256, bias=False) #in_channel = 256, out_channel = 256
        self.conv_adapt3 = conv1x1(256, 256, bias=False) #in_channel = 256, out_channel = 256
        self.conv_adapt2 = conv1x1(256, 256, bias=False) #in_channel = 256, out_channel = 256
        
        self.pre_depth = conv1x1(256, 256, groups=256, bias=False) #in_channel = 256, out_channel = 256, groups = 256
        self.depth = conv3x3(256, 1, bias=True) #in_channel = 256, out_channel = 1, bias = True

        self.pre_segm = conv1x1(256, 256, groups=256, bias=False) #in_channel = 256, out_channel = 256, groups = 256
        self.segm = conv3x3(256, num_classes, bias=True) #in_channel = 256, out_channel = num_classes
        self.relu = nn.ReLU6(inplace=True) #nn.ReLU6 is a call to the PyTorch method ReLU6 setting inplace = True

        if self.num_tasks == 3:
            self.pre_normal = conv1x1(256, 256, groups=256, bias=False) #in_channel = 256, out_channel = 256
            self.normal = conv3x3(256, 3, bias=True) #in_channel = 256, out_channel = 3, bias = True
        self._initialize_weights() #Using the object to link the initialized weights instantiated with the method
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x) # x / 2
        l3 = self.layer3(x) # 24, x / 4
        l4 = self.layer4(l3) # 32, x / 8
        l5 = self.layer5(l4) # 64, x / 16
        l6 = self.layer6(l5) # 96, x / 16
        l7 = self.layer7(l6) # 160, x / 32
        l8 = self.layer8(l7) # 320, x / 32
        l8 = self.conv8(l8)
        l7 = self.conv7(l7)
        l7 = self.relu(l8 + l7)
        l7 = self.crp4(l7)
        l7 = self.conv_adapt4(l7)
        l7 = nn.Upsample(size = l6.size()[2:], mode='bilinear', align_corners = False)(l7)

        l6 = self.conv6(l6)
        l5 = self.conv5(l5)
        l5 = self.relu(l5 + l6 + l7)
        l5 = self.crp3(l5)
        l5 = self.conv_adapt3(l5)
        l5 = nn.Upsample(size = l4.size()[2:], mode='bilinear', align_corners = False)(l5)

        l4 = self.conv4(l4)
        l4 = self.relu(l5 + l4)
        l4 = self.crp2(l4)
        l4 = self.conv_adapt2(l4)
        l4 = nn.Upsample(size = l3.size()[2:], mode='bilinear', align_corners = False)(l4)

        l3 = self.conv3(l3)
        l3 = self.relu(l3 + l4)
        l3 = self.crp1(l3)
        
        out_segm = self.pre_segm(l3)
        out_segm = self.relu(out_segm)
        out_segm = self.segm(out_segm)

        out_d = self.pre_depth(l3)
        out_d = self.relu(out_d)
        out_d = self.depth(out_d)

        if self.num_tasks == 3:
            out_n = self.pre_normal(l3)
            out_n = self.relu(out_n)
            out_n = self.normal(out_n)
            return out_segm, out_d, out_n
        else:
            return out_segm, out_d
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _make_crp(self, in_channel, out_channel, stages, groups = False):
        layers = [chained_residual_pooling(in_channel, out_channel,stages, groups = groups)]
        return nn.Sequential(*layers)
    
def network(num_classes, num_tasks):
    """Constructs the network by calling the network architecture class. This call will return the network model.
    Args:
        num_classes (int): the number of classes for the segmentation head to output.
        num_tasks (int): the number of tasks, either 2 - segm + depth, or 3 - segm + depth + surface normals
    """
    model = network_Arch(num_classes, num_tasks)
    return model