# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 21:44:34 2020

@author: abanbur
"""


import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from model import network
import cv2
import torch
from torch.autograd import Variable

cmap_Nyud = np.load('Data/cmap_nyud.npy')
cmap_Kitti = np.load('Data/cmap_kitti.npy')
depth_Coeff_Nyud = 5000. # to convert into metres
depth_Coeff_Kitti = 800.
has_Cuda = torch.cuda.is_available()
img_Scale  = 1./255
img_Mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
img_Std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
max_Depth_Nyud = 8.
min_Depth_Nyud = 0.
max_Depth_Kitti = 80.
min_Depth_Kitti = 0.
num_CLASSES = 46
num_CLASSES_NYUD = 40
num_CLASSES_KITTI = 6
num_TASKS = 2 # segm + depth

def pre_Processing(img):
    return (img * img_Scale - img_Mean) / img_Std

model_Object = network(num_classes = num_CLASSES, num_tasks = num_TASKS)
if has_Cuda:
    _ = model_Object.cuda()
_ = model_Object.eval()

check_Point = torch.load('Weights/ExpNYUDKITTI_joint.ckpt')
model_Object.load_state_dict(check_Point['state_dict'])

# NYUD
img_path = 'Examples/Example_NYUDv2_Segm_Depth/000464.png'
img_nyud = np.array(Image.open(img_path))
gt_segm_nyud = np.array(Image.open('Examples/Example_NYUDv2_Segm_Depth/segm_gt_000464.png'))

# KITTI
img_path = 'Examples/Example_KITTI_Segm_Depth/000099.png'
img_kitti = np.array(Image.open(img_path))
gt_segm_kitti = np.array(Image.open('Examples/Example_KITTI_Segm_Depth/segm_gt_000099.png'))

with torch.no_grad():
    # nyud
    img_var = Variable(torch.from_numpy(pre_Processing(img_nyud).transpose(2, 0, 1)[None]), requires_grad=False).float()
    if has_Cuda:
        img_var = img_var.cuda()
    segm, depth = model_Object(img_var)
    segm = cv2.resize(segm[0, :(num_CLASSES_NYUD)].cpu().data.numpy().transpose(1, 2, 0), img_nyud.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
    depth = cv2.resize(depth[0, 0].cpu().data.numpy(), img_nyud.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
    segm_nyud = cmap_Nyud[segm.argmax(axis=2) + 1].astype(np.uint8)
    depth_nyud = np.abs(depth)
    # kitti
    img_var = Variable(torch.from_numpy(pre_Processing(img_kitti).transpose(2, 0, 1)[None]), requires_grad=False).float()
    if has_Cuda:
        img_var = img_var.cuda()
    segm, depth = model_Object(img_var)
    segm = cv2.resize(segm[0, (num_CLASSES_NYUD):(num_CLASSES_NYUD + num_CLASSES_KITTI)].cpu().data.numpy().transpose(1, 2, 0), img_kitti.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
    depth = cv2.resize(depth[0, 0].cpu().data.numpy(), img_kitti.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
    segm_kitti = cmap_Kitti[segm.argmax(axis=2)].astype(np.uint8)
    depth_kitti = np.abs(depth)

plt.figure(figsize=(18, 12))
plt.subplot(141)
plt.imshow(img_nyud)
plt.title('NYUD: img')
plt.axis('off')
plt.subplot(142)
plt.imshow(cmap_Nyud[gt_segm_nyud + 1])
plt.title('NYUD: gt segm')
plt.axis('off')
plt.subplot(143)
plt.imshow(segm_nyud)
plt.title('NYUD: pred segm')
plt.axis('off')
plt.subplot(144)
plt.imshow(depth_nyud, cmap='plasma', vmin=min_Depth_Nyud, vmax=max_Depth_Nyud)
plt.title('NYUD: pred depth')
plt.axis('off')
plt.figure(figsize=(18,12))
plt.subplot(141)
plt.imshow(img_kitti)
plt.title('KITTI: img')
plt.axis('off')
plt.subplot(142)
plt.imshow(gt_segm_kitti)
plt.title('KITTI: gt segm')
plt.axis('off')
plt.subplot(143)
plt.imshow(segm_kitti)
plt.title('KITTI: pred segm')
plt.axis('off')
plt.subplot(144)
plt.imshow(depth_kitti, cmap='plasma', vmin=min_Depth_Kitti, vmax=max_Depth_Kitti)
plt.title('KITTI: pred depth')
plt.axis('off');




