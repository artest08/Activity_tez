#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:20:52 2019

@author: esat
"""

import os
import sys
import numpy as np
import math
import cv2
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

sys.path.insert(0, "../../")
import video_transforms

soft=nn.Softmax(dim=1)
def VideoSpatialPrediction_lstm(
        vid_name,
        net,
        num_categories,
        start_frame=0,
        num_frames=0,
        temporal_length=16
        ):

    if num_frames == 0:
        imglist = os.listdir(vid_name)
        newImageList=[]
        for item in imglist:
            if 'img' in item:
               newImageList.append(item) 
        duration = len(newImageList)
    else:
        duration = num_frames

    clip_mean = [0.485, 0.456, 0.406]
    clip_std = [0.229, 0.224, 0.225]
    normalize = video_transforms.Normalize(mean=clip_mean,
                                     std=clip_std)
    val_transform = video_transforms.Compose([
            video_transforms.ToTensor(),
            normalize,
        ])

    # selection
    #step = int(math.floor((duration-1)/(num_samples-1)))
    dims = (224,224,3,duration,10)
    rgb = np.zeros(shape=dims, dtype=np.float64)
    dims = (256,340,3,duration)

    for i in range(duration):
        img_file = os.path.join(vid_name, 'img_{0:05d}.jpg'.format(i+1))
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, dims[1::-1])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flip = img[:,::-1,:].copy()
        rgb[:,:,:,i,0] = img[16:240, 60:284, :]
        rgb[:,:,:,i,1] = img[:224, :224, :]
        rgb[:,:,:,i,2] = img[:224, -224:, :]
        rgb[:,:,:,i,3] = img[-224:, :224, :]
        rgb[:,:,:,i,4] = img[-224:, -224:, :]
        rgb[:,:,:,i,5] = img_flip[16:240, 60:284, :]
        rgb[:,:,:,i,6] = img_flip[:224, :224, :]
        rgb[:,:,:,i,7] = img_flip[:224, -224:, :]
        rgb[:,:,:,i,8] = img_flip[-224:, :224, :]
        rgb[:,:,:,i,9] = img_flip[-224:, -224:, :]

    # crop
    

    rgb_list = []
    half_temporal_length=int(temporal_length/2)
    for i in range(10):
        for c_index in range(half_temporal_length):
            cur_img = rgb[:,:,:,c_index,i].squeeze()
            cur_img_tensor = val_transform(cur_img)
            rgb_list.append(np.expand_dims(cur_img_tensor.numpy(), 0))
            
        for c_indexStart in range(half_temporal_length,duration-np.mod(duration,half_temporal_length)-half_temporal_length,half_temporal_length):
            for c_index in range(c_indexStart,c_indexStart+half_temporal_length):
                cur_img = rgb[:,:,:,c_index,i].squeeze()
                cur_img_tensor = val_transform(cur_img)
                rgb_list.append(np.expand_dims(cur_img_tensor.numpy(), 0))
            for c_index in range(c_indexStart,c_indexStart+half_temporal_length):
                cur_img = rgb[:,:,:,c_index,i].squeeze()
                cur_img_tensor = val_transform(cur_img)
                rgb_list.append(np.expand_dims(cur_img_tensor.numpy(), 0))
                
        for c_index in range(duration-np.mod(duration,half_temporal_length)-half_temporal_length,duration-np.mod(duration,half_temporal_length)):
            cur_img = rgb[:,:,:,c_index,i].squeeze()
            cur_img_tensor = val_transform(cur_img)
            rgb_list.append(np.expand_dims(cur_img_tensor.numpy(), 0))
            
    sample_number=len(rgb_list)
    rgb_np = np.concatenate(rgb_list,axis=0)
    # print(rgb_np.shape)
    batch_size = 256
    prediction = np.zeros((num_categories,sample_number))
    num_batches = int(math.ceil(sample_number/batch_size))
    with torch.no_grad():
        for bb in range(num_batches):
            span = range(batch_size*bb, min(sample_number,batch_size*(bb+1)))
            input_data = rgb_np[span,:,:,:]
            imgDataTensor = torch.from_numpy(input_data).type(torch.FloatTensor).cuda()
            output = net(imgDataTensor)
    #        outputSoftmax=soft(output)
            result = output.data.cpu().numpy()
            prediction[:, span] = np.transpose(result)

    return prediction
