#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:02:52 2019

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
def VideoTemporalPrediction_bert(
        vid_name,
        net,
        num_categories,
        start_frame=0,
        num_frames=0,
        num_seg=16,
        length = 1
        ):

    if num_frames == 0:
        imglist = os.listdir(vid_name)
        newImageList=[]
        for item in imglist:
            if 'flow_x' in item:
               newImageList.append(item) 
        duration = len(newImageList)
    else:
        duration = num_frames

    clip_mean = [0.5] * 2
    clip_std = [0.226] * 2
    normalize = video_transforms.Normalize(mean=clip_mean,
                                     std=clip_std)
    
    val_transform = video_transforms.Compose([
            video_transforms.ToTensor(),
            normalize,
        ])

    # selection
    #step = int(math.floor((duration-1)/(num_samples-1)))
    dims = (224,224,duration)
    dims = (256,340,duration)
    average_duration = int(duration / num_seg)
    offsetMainIndexes = []
    offsets = []
    for seg_id in range(num_seg):
        if average_duration >= length:
            offsetMainIndexes.append(int((average_duration - length + 1)/2 + seg_id * average_duration))
        elif duration >=length:
            average_part_length = int(np.floor((duration-length)/num_seg))
            offsetMainIndexes.append(int((average_part_length*(seg_id) + average_part_length*(seg_id+1))/2))
#        else:
#            offsetMainIndexes.append(0)
    for mainOffsetValue in offsetMainIndexes:
        for lengthID in range(1, length+1):
             offsets.append(lengthID + mainOffsetValue)
    imageList=[]
    imageList1=[]
    imageList2=[]
    imageList3=[]
    imageList4=[]    
    imageList5=[]  
    imageList6=[]
    imageList7=[]
    imageList8=[]
    imageList9=[]    
    imageList10=[]  
    interpolation = cv2.INTER_LINEAR
    
    for index in offsets:
        if 'ucf101' in vid_name or 'window' in vid_name:
            flow_x_file = os.path.join(vid_name, 'flow_x_{0:05d}.jpg'.format(index))
            flow_y_file = os.path.join(vid_name, 'flow_y_{0:05d}.jpg'.format(index))
        elif 'hmdb51' in vid_name:
            flow_x_file = os.path.join(vid_name, 'flow_x_{0:05d}'.format(index))
            flow_y_file = os.path.join(vid_name, 'flow_y_{0:05d}'.format(index))
        img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
        img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
        img_x = np.expand_dims(img_x,-1)
        img_y = np.expand_dims(img_y,-1)
        img = np.concatenate((img_x,img_y),2)    
        img = cv2.resize(img, dims[1::-1],interpolation)
        img_flip = img[:,::-1,:].copy()
        imageList1.append(img[16:240, 60:284, :])
        imageList2.append(img[:224, :224, :])
        imageList3.append(img[:224, -224:, :])
        imageList4.append(img[-224:, :224, :])
        imageList5.append(img[-224:, -224:, :])
        imageList6.append(img_flip[16:240, 60:284, :])
        imageList7.append(img_flip[:224, :224, :])
        imageList8.append(img_flip[:224, -224:, :])
        imageList9.append(img_flip[-224:, :224, :])
        imageList10.append(img_flip[-224:, -224:, :])


    imageList=imageList1+imageList2+imageList3+imageList4+imageList5+imageList6+imageList7+imageList8+imageList9+imageList10
    
    rgb_list=[]     

    for i in range(len(imageList)):
        cur_img = imageList[i]
        cur_img_tensor = val_transform(cur_img)
        rgb_list.append(np.expand_dims(cur_img_tensor.numpy(), 0))
         
    input_data=np.concatenate(rgb_list,axis=0)   
    with torch.no_grad():
        imgDataTensor = torch.from_numpy(input_data).type(torch.FloatTensor).cuda()
        imgDataTensor = imgDataTensor.view(-1,length*2,224,224)
        output,_,_,_ = net(imgDataTensor)
#        outputSoftmax=soft(output)
        result = output.data.cpu().numpy()
        mean_result=np.mean(result,0)
        prediction=np.argmax(mean_result)
        
    return prediction, mean_result