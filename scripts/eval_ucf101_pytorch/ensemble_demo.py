#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:18:06 2020

@author: esat
"""

import os, sys
import collections
import numpy as np
import cv2
import math
import random
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from numpy import linalg as LA

from sklearn.metrics import confusion_matrix

datasetFolder="../../datasets"
sys.path.insert(0, "../../")
import models
from VideoSpatialPrediction3D import VideoSpatialPrediction3D
from VideoSpatialPrediction3D_bert import VideoSpatialPrediction3D_bert

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition RGB Test Case')

parser.add_argument('--dataset', '-d', default='hmdb51',
                    choices=["ucf101", "hmdb51"],
                    help='dataset: ucf101 | hmdb51')
parser.add_argument('--arch1', '-a', metavar='ARCH', default='rgb_r2plus1d_32f_34',
                    choices=model_names)
parser.add_argument('--arch2', '-b', metavar='ARCH', default='rgb_r2plus1d_32f_34_bert10',
                    choices=model_names)
parser.add_argument('--arch3', '-c', metavar='ARCH', default='rgb_resneXt3D64f101_student_mars',
                    choices=model_names)
parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')

parser.add_argument('-w', '--window', default=3, type=int, metavar='V',
                    help='validation file index (default: 3)')

parser.add_argument('-t', '--tsn', dest='tsn', action='store_true',
                    help='TSN Mode')

parser.add_argument('-v', '--val', dest='window_val', action='store_true',
                    help='Window Validation Selection')

multiGPUTest=False
multiGPUTrain=False

ten_crop_enabled = False
multiple_clips_enabled = True

third_enabled = False

num_seg_rgb=16
num_seg_pose=16
num_seg_flow=16
len_flow=1
poseEnabled = True
num_seg_3D = 1
length_3D = 64

def length_selection(architecture_name):
    if '64f' in architecture_name:
        arch_length=64
    elif '32f' in architecture_name:
        arch_length=32
    elif '8f' in architecture_name:
        arch_length=8    
    else:
        arch_length=16   
    return arch_length

def extension_selection(architecture_name, dataset):
    if 'rgb' in architecture_name:
        extension = 'img_{0:05d}.jpg'
    elif 'pose' in architecture_name:
        extension = 'img_{0:05d}.jpg'
    elif 'flow' in architecture_name:
        if 'ucf101' in args.dataset or 'window' in args.dataset:
            extension = 'flow_{0}_{1:05d}.jpg'
        elif 'hmdb51' in args.dataset:
            extension = 'flow_{0}_{1:05d}'
 
    return extension

def buildModel(model_path,arch,num_categories):
    global multiGPUTrain
    if 'rgb' in arch:
        model=models.__dict__[arch](modelPath='', num_classes=num_categories,length=num_seg_rgb)
    elif 'flow' in arch:
        model=models.__dict__[arch](modelPath='', num_classes=num_categories,length=num_seg_flow)
    elif 'pose' in arch:
        multiGPUTrain = True
        model=models.__dict__[arch](modelPath='', num_classes=num_categories,length=num_seg_pose)
    params = torch.load(model_path)
        
    if args.tsn:
        new_dict = {k[7:]: v for k, v in params['state_dict'].items()} 
        model_dict=model.state_dict() 
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
    elif multiGPUTest:
        model=torch.nn.DataParallel(model)
        new_dict={"module."+k: v for k, v in params['state_dict'].items()} 
        model.load_state_dict(new_dict)
    elif multiGPUTrain:
        new_dict = {k[7:]: v for k, v in params['state_dict'].items()} 
        model_dict=model.state_dict() 
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(params['state_dict'])
    model.cuda()
    model.eval()  
    return model

def main():
    global args
    args = parser.parse_args()
    
    if args.tsn:    
        modelLocationRGB="./checkpoint/"+args.dataset+"_tsn_"+args.arch_rgb+"_split"+str(args.split)
        modelLocationFlow="./checkpoint/"+args.dataset+"_tsn_"+args.arch_flow+"_split"+str(args.split)
    else:
        modelLocationARCH1="./checkpoint/"+args.dataset+"_"+args.arch1+"_split"+str(args.split)
        modelLocationARCH2="./checkpoint/"+args.dataset+"_"+args.arch2+"_split"+str(args.split)
        modelLocationARCH3="./checkpoint/"+args.dataset+"_"+args.arch3+"_split"+str(args.split)

    model_path_arch1 = os.path.join('../../',modelLocationARCH1,'model_best.pth.tar') 
    model_path_arch2 = os.path.join('../../',modelLocationARCH2,'model_best.pth.tar') 
    model_path_arch3 = os.path.join('../../',modelLocationARCH3,'model_best.pth.tar') 
    
    if args.dataset=='ucf101':
        frameFolderName = "ucf101_frames"
    elif args.dataset=='hmdb51':
        frameFolderName = "hmdb51_frames"
    elif args.dataset=='window':
        frameFolderName = "window_frames"
    data_dir=os.path.join(datasetFolder,frameFolderName)
    
    arch1_length = length_selection(args.arch1) 
    arch2_length = length_selection(args.arch2) 
    arch3_length = length_selection(args.arch3) 
        

    if args.window_val:
        val_fileName = "window%d.txt" %(args.window)
    else:
        val_fileName = "val_flow_split%d.txt" %(args.split)
       
    arch1_extension = extension_selection(args.arch1, args.dataset)
    arch2_extension = extension_selection(args.arch2, args.dataset)
    arch3_extension = extension_selection(args.arch3, args.dataset)


    val_file=os.path.join(datasetFolder,'settings',args.dataset,val_fileName)
    
    start_frame = 0
    if args.dataset=='ucf101':
        num_categories = 101
    elif args.dataset=='hmdb51':
        num_categories = 51
    elif args.dataset=='window':
        num_categories = 3

    model_start_time = time.time()
    arch1_net = buildModel(model_path_arch1,args.arch1,num_categories)
    arch2_net = buildModel(model_path_arch2,args.arch2,num_categories)
    arch3_net = buildModel(model_path_arch3,args.arch3,num_categories)
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print("Action recognition model is loaded in %4.4f seconds." % (model_time))

    
    f_val = open(val_file, "r")
    val_list = f_val.readlines()
    print("we got %d test videos" % len(val_list))

    line_id = 1
    match_count = 0
    match_count_top3 = 0

    y_true=[]
    y_pred=[]
    timeList=[]
    #result_list = []
    for line in val_list:
        line_info = line.split(" ")
        clip_path = os.path.join(data_dir,line_info[0])
        duration = int(line_info[1])
        input_video_label = int(line_info[2]) 
        
        start = time.time()
        
        if not multiple_clips_enabled:
            _ , arch1_result, _ = VideoSpatialPrediction3D_bert(
                                           clip_path,
                                           arch1_net,
                                           num_categories,
                                           args.arch1,
                                           start_frame,
                                           duration,
                                           num_seg=num_seg_3D ,
                                           length = arch1_length, 
                                           extension = arch1_extension,
                                           ten_crop = ten_crop_enabled)
            
            _ , arch2_result, _ = VideoSpatialPrediction3D_bert(
                                           clip_path,
                                           arch2_net,
                                           num_categories,
                                           args.arch2,
                                           start_frame,
                                           duration,
                                           num_seg=num_seg_3D ,
                                           length = arch2_length, 
                                           extension = arch2_extension,
                                           ten_crop = ten_crop_enabled)
            
            if third_enabled:
                _ , arch3_result, _ = VideoSpatialPrediction3D_bert(
                                               clip_path,
                                               arch3_net,
                                               num_categories,
                                               args.arch3,
                                               start_frame,
                                               duration,
                                               num_seg=num_seg_3D ,
                                               length = arch3_length, 
                                               extension = arch3_extension,
                                               ten_crop = ten_crop_enabled)    
    
    
        else:
            _ , arch1_result, _ = VideoSpatialPrediction3D(
                                           clip_path,
                                           arch1_net,
                                           num_categories,
                                           args.arch1,
                                           start_frame,
                                           duration,
                                           length = arch1_length, 
                                           extension = arch1_extension,
                                           ten_crop = ten_crop_enabled)
            
            _ , arch2_result, _ = VideoSpatialPrediction3D(
                                           clip_path,
                                           arch2_net,
                                           num_categories,
                                           args.arch2,
                                           start_frame,
                                           duration,
                                           length = arch2_length, 
                                           extension = arch2_extension,
                                           ten_crop = ten_crop_enabled)
            if third_enabled:
                _ , arch3_result, _ = VideoSpatialPrediction3D(
                                               clip_path,
                                               arch3_net,
                                               num_categories,
                                               args.arch3,
                                               start_frame,
                                               duration,
                                               length = arch3_length, 
                                               extension = arch3_extension,
                                               ten_crop = ten_crop_enabled)
                         
            
        end = time.time()
        estimatedTime=end-start
        timeList.append(estimatedTime)
        
        arch1_result = arch1_result / LA.norm(arch1_result)
        arch2_result = arch2_result / LA.norm(arch2_result)
        
        if third_enabled:
            arch3_result = arch3_result / LA.norm(arch3_result)
            combined_result = arch1_result + arch2_result + arch3_result
        else:
            combined_result = arch1_result + arch2_result
        pred_index = np.argmax(combined_result)
        top3 = combined_result.argsort()[::-1][:3]
        
        print("Sample %d/%d: GT: %d, Prediction: %d" % (line_id, len(val_list), input_video_label, pred_index))
        print("Estimated Time  %0.4f" % estimatedTime)
        print("------------------")
        if pred_index == input_video_label:
            match_count += 1
        if input_video_label in top3:
            match_count_top3 += 1

        line_id += 1
        y_true.append(input_video_label)
        y_pred.append(pred_index)

        
    print(confusion_matrix(y_true,y_pred))

    print("Accuracy with mean calculation is %4.4f" % (float(match_count)/len(val_list)))
    print("top3 accuracy %4.4f" % (float(match_count_top3)/len(val_list)))
    print(modelLocationARCH1)
    print(modelLocationARCH2)
    if third_enabled:
        print(modelLocationARCH3)
    print("Mean Estimated Time %0.4f" % (np.mean(timeList)))  
    if multiple_clips_enabled:
        print('multiple clips')
    else:
        print('one clips')
    if ten_crop_enabled:
        print('10 crops')
    else:
        print('single crop')
    
#    resultDict={'y_true':y_true,'y_pred':y_pred}
    
#    np.save('results/%s.npy' %(args.dataset+'_'+args.arch_rgb+'_'+ args.arch_flow +"_split"+str(args.split)), resultDict) 

if __name__ == "__main__":
    main()

