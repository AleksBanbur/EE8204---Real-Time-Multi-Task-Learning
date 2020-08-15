# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:20:05 2020

@author: abanbur
"""


import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from model import network
import cv2
import torch
from torch.autograd import Variable

color_Map = np.load('Data/cmap_kitti.npy')
depth_Coeff = 800. # Converts into meters
has_Cuda = torch.cuda.is_available()
img_Scale  = 1./255
img_Mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
img_Std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
max_Depth = 80.
min_Depth = 0.
num_CLASSES = 6
num_TASKS = 2 # segm + depth

def pre_Processing(img):
    return (img * img_Scale - img_Mean) / img_Std

model_Object = network(num_classes=num_CLASSES, num_tasks=num_TASKS)
if has_Cuda:
    _ = model_Object.cuda()
_ = model_Object.eval()

check_Point = torch.load('Weights/ExpKITTI_joint.ckpt')
model_Object.load_state_dict(check_Point['state_dict'])

img_Path = 'Examples/Example_KITTI_Segm_Depth/000099.png'
img = np.array(Image.open(img_Path))
gt_segm = np.array(Image.open('Examples\Example_KITTI_Segm_Depth\segm_gt_000099.png'))

with torch.no_grad():
    img_var = Variable(torch.from_numpy(pre_Processing(img).transpose(2, 0, 1)[None]), requires_grad = False).float()
    if has_Cuda:
        img_var = img_var.cuda()
    segm, depth = model_Object(img_var)
    segm = cv2.resize(segm[0, :num_CLASSES].cpu().data.numpy().transpose(1, 2, 0), img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
    depth = cv2.resize(depth[0, 0].cpu().data.numpy(), img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
    segm = color_Map[segm.argmax(axis=2)].astype(np.uint8)
    depth = np.abs(depth)
plt.figure(figsize=(18, 12))
plt.subplot(141)
plt.imshow(img)
plt.title('orig img')
plt.axis('off')
plt.subplot(142)
plt.imshow(gt_segm)
plt.title('gt segm')
plt.axis('off')
plt.subplot(143)
plt.imshow(segm)
plt.title('pred segm')
plt.axis('off')
plt.subplot(144)
plt.imshow(depth, cmap='plasma', vmin=min_Depth, vmax=max_Depth)
plt.title('pred depth')
plt.axis('off');
