#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:16:39 2019

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
def VideoSpatialPrediction3D(
        vid_name,
        net,
        num_categories,
        architecture_name,
        start_frame=0,
        num_frames=0,
        length = 16,
        extension = 'img_{0:05d}.jpg'
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
    
    if 'I3D' in architecture_name:
        scale = 1
    else:
        scale = 0.5
        
    if scale == 0.5:
        clip_mean = [114.7748, 107.7354, 99.4750]
        clip_std = [1, 1, 1]
        normalize = video_transforms.Normalize(mean=clip_mean,
                                 std=clip_std)
        val_transform = video_transforms.Compose([
                video_transforms.ToTensor2(),
                normalize,
            ])

    elif scale == 1:
        if not 'resnet' in architecture_name:
            clip_mean = [0.5, 0.5, 0.5] 
            clip_std = [0.5, 0.5, 0.5]
        else:
            clip_mean = [0.45, 0.45, 0.45]
            clip_std = [0.225, 0.225, 0.225] 
        normalize = video_transforms.Normalize(mean=clip_mean,
                                 std=clip_std)
        val_transform = video_transforms.Compose([
                video_transforms.ToTensor(),
                normalize,
            ])
    


    # selection
    #step = int(math.floor((duration-1)/(num_samples-1)))
    dims2 = (224,224,3,duration)
    
    imageSize=int(224 * scale)
    dims = (int(256 * scale),int(340 * scale),3,duration)
    duration = duration - 1
    
    offsets = []
    
    offsetMainIndexes = list(range(1,duration-length,length))
    if len(offsetMainIndexes) == 0:
        offsets = list(range(1,duration+2))*int(np.floor(length/(duration+1))) + list(range(1,length%(duration+1)+1))
    else:
        for mainOffsetValue in offsetMainIndexes:
            for lengthID in range(1, length+1):
                 offsets.append(lengthID + mainOffsetValue)
                 
#    offsetMainIndexes = list(range(0,duration,length))
#    for mainOffsetValue in offsetMainIndexes:
#        for lengthID in range(1, length+1):
#            loaded_frame_index = lengthID + mainOffsetValue
#            moded_loaded_frame_index = loaded_frame_index % (duration + 1)
#            if moded_loaded_frame_index == 0:
#                moded_loaded_frame_index = (duration + 1)
#            offsets.append(moded_loaded_frame_index)
             
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
    imageList11=[] 
    imageList12=[] 
    interpolation = cv2.INTER_LINEAR
    
    for index in offsets:
        img_file = os.path.join(vid_name, extension.format(index))
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)

        img = cv2.resize(img, dims[1::-1],interpolation)

        #img2 = cv2.resize(img, dims2[1::-1],interpolation)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flip = img[:,::-1,:].copy()
        #img_flip2 = img2[:,::-1,:].copy()
        imageList1.append(img[int(16 * scale):int(16 * scale + imageSize), int(58 * scale) : int(58 * scale + imageSize), :])
        imageList2.append(img[:imageSize, :imageSize, :])
        imageList3.append(img[:imageSize, -imageSize:, :])
        imageList4.append(img[-imageSize:, :imageSize, :])
        imageList5.append(img[-imageSize:, -imageSize:, :])
        imageList6.append(img_flip[int(16 * scale):int(16 * scale + imageSize), int(58 * scale) : int(58 * scale + imageSize), :])
        imageList7.append(img_flip[:imageSize, :imageSize, :])
        imageList8.append(img_flip[:imageSize, -imageSize:, :])
        imageList9.append(img_flip[-imageSize:, :imageSize, :])
        imageList10.append(img_flip[-imageSize:, -imageSize:, :])
#        imageList11.append(img2)
#        imageList12.append(img_flip2)


    imageList=imageList1+imageList2+imageList3+imageList4+imageList5+imageList6+imageList7+imageList8+imageList9+imageList10
    
    #imageList=imageList1
    
    #imageList=imageList11+imageList12
    
    rgb_list=[]     

    for i in range(len(imageList)):
        cur_img = imageList[i]
        cur_img_tensor = val_transform(cur_img)
        rgb_list.append(np.expand_dims(cur_img_tensor.numpy(), 0))
         
    input_data=np.concatenate(rgb_list,axis=0)   

    batch_size = length
    sample_size = int(batch_size/length)
    result = np.zeros((int(input_data.shape[0]/length),num_categories))
    num_batches = int(math.ceil(float(input_data.shape[0])/batch_size))

    with torch.no_grad():
        for bb in range(num_batches):
            span = range(batch_size*bb, min(input_data.shape[0],batch_size*(bb+1)))
            input_data_batched = input_data[span,:,:,:]
            imgDataTensor = torch.from_numpy(input_data_batched).type(torch.FloatTensor).cuda()
            imgDataTensor = imgDataTensor.view(-1,length,3,imageSize,imageSize).transpose(1,2)
            output = net(imgDataTensor)
            #output,_,_,_ = net(imgDataTensor)
            span = range(sample_size*bb, min(int(input_data.shape[0]/64),sample_size*(bb+1)))
            result[span,:] = output.data.cpu().numpy()
        mean_result=np.mean(result,0)
        prediction=np.argmax(mean_result)
        
    return prediction, mean_result