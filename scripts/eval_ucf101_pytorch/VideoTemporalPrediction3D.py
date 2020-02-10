#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:34:06 2019

@author: esat
"""

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
def VideoTemporalPrediction3D(
        vid_name,
        net,
        num_categories,
        start_frame=0,
        num_frames=0,
        num_seg=4,
        length = 16,
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
    clip_std = [0.5] * 2
    normalize = video_transforms.Normalize(mean=clip_mean,
                                     std=clip_std)
    
    val_transform = video_transforms.Compose([
            video_transforms.ToTensor(),
            normalize,
        ])

    # selection
    scale = 1
    imageSize=int(224 * scale)
    dims = (int(256 * scale),int(340 * scale),3,duration)
    duration = duration - 1
    average_duration = int(duration / num_seg)
    offsetMainIndexes = []
    offsets = []
    for seg_id in range(num_seg):
        if average_duration >= length:
            offsetMainIndexes.append(int((average_duration - length + 1)/2 + seg_id * average_duration))
        elif duration >=length:
            average_part_length = int(np.floor((duration-length)/num_seg))
            offsetMainIndexes.append(int((average_part_length*(seg_id) + average_part_length*(seg_id+1))/2))
        else:
            increase = int(duration / num_seg)
            offsetMainIndexes.append(0 + seg_id * increase)
    for mainOffsetValue in offsetMainIndexes:
        for lengthID in range(1, length+1):
            loaded_frame_index = lengthID + mainOffsetValue
            moded_loaded_frame_index = loaded_frame_index % (duration + 1)
            if moded_loaded_frame_index == 0:
                moded_loaded_frame_index = (duration + 1)
            offsets.append(moded_loaded_frame_index)
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
        if 'ucf101' or 'window' in vid_name:
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


    imageList=imageList1+imageList2+imageList3+imageList4+imageList5+imageList6+imageList7+imageList8+imageList9+imageList10
    #imageList=imageList1
    rgb_list=[]     

    for i in range(len(imageList)):
        cur_img = imageList[i]
        cur_img_tensor = val_transform(cur_img)
        rgb_list.append(np.expand_dims(cur_img_tensor.numpy(), 0))
         
    input_data=np.concatenate(rgb_list,axis=0)   

    batch_size = 64
    sample_size = int(batch_size/length)
    result = np.zeros((int(input_data.shape[0]/64),num_categories))
    num_batches = int(math.ceil(float(input_data.shape[0])/batch_size))

    with torch.no_grad():
        for bb in range(num_batches):
            span = range(batch_size*bb, min(input_data.shape[0],batch_size*(bb+1)))
            input_data_batched = input_data[span,:,:,:]
            imgDataTensor = torch.from_numpy(input_data_batched).type(torch.FloatTensor).cuda()
            imgDataTensor = imgDataTensor.view(-1,length,2,imageSize,imageSize).transpose(1,2)
     #       output = net(imgDataTensor)
            output,_,_,_ = net(imgDataTensor)
    #        outputSoftmax=soft(output)
            span = range(sample_size*bb, min(int(input_data.shape[0]/64),sample_size*(bb+1)))
            result[span,:] = output.data.cpu().numpy()
        mean_result=np.mean(result,0)
        prediction=np.argmax(mean_result)
        
    return prediction, mean_result