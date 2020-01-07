#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:52:03 2019

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
import csv
from sklearn.metrics import confusion_matrix

datasetFolder="../../datasets"
sys.path.insert(0, "../../")
import models
from VideoSpatialPrediction_bert import VideoSpatialPrediction_bert
from VideoSpatialPrediction3D_bert import VideoSpatialPrediction3D_bert

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition RGB Test Case')

parser.add_argument('--dataset', '-d', default='smtV2',
                    choices=["ucf101", "hmdb51"],
                    help='dataset: ucf101 | hmdb51')
parser.add_argument('--arch', '-a', metavar='ARCH', default='rgb_resneXt3D64f101_bert10XY',
                    choices=model_names)

parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')
parser.add_argument('-t', '--tsn', dest='tsn', action='store_true',
                    help='TSN Mode')
multiGPUTest=False
num_seg=16
num_seg_3D=1

result_dict = {}

def buildModel(model_path,num_categories):
    if not '3D' in args.arch:
        model=models.__dict__[args.arch](modelPath='', num_classes=num_categories,length=num_seg)
    else:
        model=models.__dict__[args.arch](modelPath='', num_classes=num_categories,length=num_seg_3D)
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
    else:
        model.load_state_dict(params['state_dict'])
    model.cuda()
    model.eval()  
    return model


def main():
    global args
    args = parser.parse_args()
    if '64' in args.arch:
        length=64
    else:
        length=16
        
    if args.tsn:    
        modelLocation="./checkpoint/"+args.dataset+"_tsn_"+args.arch+"_split"+str(args.split)
    else:
        modelLocation="./checkpoint/"+args.dataset+"_"+args.arch+"_split"+str(args.split)

    model_path = os.path.join('../../',modelLocation,'model_best.pth.tar') 
    
    if args.dataset=='ucf101':
        frameFolderName = "ucf101_frames"
    elif args.dataset=='hmdb51':
        frameFolderName = "hmdb51_frames"
    elif args.dataset=='smtV2':
        frameFolderName = "smtV2_frames"
    data_dir=os.path.join(datasetFolder,frameFolderName)
    

    val_fileName = "test_rgb_split%d.txt" %(args.split)
    if 'rgb' in args.arch:
        extension = 'img_{0:05d}.jpg'
    elif 'pose' in args.arch:
        extension = 'pose1_{0:05d}.jpg'

    val_file=os.path.join(datasetFolder,'settings',args.dataset,val_fileName)
    
    start_frame = 0
    if args.dataset=='ucf101':
        num_categories = 101
    elif args.dataset=='hmdb51':
        num_categories = 51
    elif args.dataset=='smtV2':
        num_categories = 174
        

    model_start_time = time.time()
    spatial_net=buildModel(model_path,num_categories)
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print("Action recognition model is loaded in %4.4f seconds." % (model_time))

    
    f_val = open(val_file, "r")
    val_list = f_val.readlines()
    print("we got %d test videos" % len(val_list))

    line_id = 1
    match_count = 0

    y_true=[]
    y_pred=[]
    timeList=[]
    #result_list = []
    with open('%s.csv' %(args.arch), mode='w') as result_csvfile:
        employee_writer = csv.writer(result_csvfile, delimiter=';')    
        for i,line in enumerate(val_list):
            line_info = line.split(" ")
            clip_path = os.path.join(data_dir,line_info[0])
            duration = int(line_info[1])
            input_video_label = int(line_info[2]) 
            
            start = time.time()
            if not '3D' in args.arch:
                spatial_prediction = VideoSpatialPrediction_bert(
                        clip_path,
                        spatial_net,
                        num_categories,
                        start_frame,
                        duration,
                        num_seg=num_seg,
                        extension = extension)
            else:
                spatial_prediction = VideoSpatialPrediction3D_bert(
                    clip_path,
                    spatial_net,
                    num_categories,
                    start_frame,
                    duration,
                    num_seg=num_seg_3D ,
                    length = length, 
                    extension = extension)
                
            
            end = time.time()
            estimatedTime=end-start
            timeList.append(estimatedTime)
            
            pred_index, mean_result = spatial_prediction
            if args.dataset=='smtV2':
                top5 = np.argsort(mean_result)[::-1][:5]                   
                employee_writer.writerow([line_info[0], str(top5[0]), str(top5[1]),str(top5[2]),str(top5[3]),str(top5[4])])
                
            print("Sample %d/%d: GT: %d, Prediction: %d" % (line_id, len(val_list), input_video_label, pred_index))
            print("Estimated Time  %0.4f" % estimatedTime)
            print("------------------")
            if pred_index == input_video_label:
                match_count += 1
    
            line_id += 1
            y_true.append(input_video_label)
            y_pred.append(pred_index)
    
            
        print(confusion_matrix(y_true,y_pred))
    
        print("Accuracy with mean calculation is %4.4f" % (float(match_count)/len(val_list)))
        print(modelLocation)
        print("Mean Estimated Time %0.4f" % (np.mean(timeList)))  
        
        resultDict={'y_true':y_true,'y_pred':y_pred}
        
        np.save('results/%s.npy' %(args.dataset+args.arch+"_split"+str(args.split)), resultDict) 

if __name__ == "__main__":
    main()

