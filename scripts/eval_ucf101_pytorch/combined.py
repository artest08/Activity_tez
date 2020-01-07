#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 12:46:46 2018

@author: esat
"""


import os
import sys
import collections
import numpy as np
import cv2
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

sys.path.insert(0, "../../")
import models

from VideoTemporalPrediction import VideoTemporalPrediction
from VideoSpatialPrediction import VideoSpatialPrediction

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def main():

    model_path_temporal = '../../checkpoints_split2-tsn-flow/model_best.pth.tar'
    model_path_spatial = '../../checkpoints_split2-tsn-rgb/model_best.pth.tar'
    data_dir = "../../datasets/ucf101_frames"
    start_frame = 0
    num_categories = 101

    model_start_time = time.time()
    params = torch.load(model_path_temporal)
    temporal_net = models.flow_resnet152(pretrained=False, num_classes=101)
    new_dict = {k[7:]: v for k, v in params['state_dict'].items()} 
    model_dict=temporal_net.state_dict() 
    model_dict.update(new_dict)
    temporal_net.load_state_dict(model_dict)
    temporal_net.cuda()
    temporal_net.eval()
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print("Action recognition temporal model is loaded in %4.4f seconds." % (model_time))
    
    model_start_time = time.time()
    params = torch.load(model_path_spatial)
    spatial_net = models.rgb_resnet152(pretrained=False, num_classes=101)
    new_dict = {k[7:]: v for k, v in params['state_dict'].items()} 
    model_dict=spatial_net.state_dict() 
    model_dict.update(new_dict)
    spatial_net.load_state_dict(model_dict)
    spatial_net.cuda()
    spatial_net.eval()
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print("Action recognition RGB model is loaded in %4.4f seconds." % (model_time))

    val_file = "val_rgb_split2.txt"
    f_val = open(val_file, "r")
    val_list = f_val.readlines()
    print("we got %d test videos" % len(val_list))

    line_id = 1
    match_count = 0
    result_list = []

    for line in val_list:
        line_info = line.split(" ")
        clip_path = os.path.join(data_dir,line_info[0])
        duration = int(line_info[1])
        input_video_label = int(line_info[2]) 

        temporal_prediction = VideoTemporalPrediction(
                clip_path,
                temporal_net,
                num_categories,
                start_frame,
                duration)
        
        spatial_prediction = VideoSpatialPrediction(
                clip_path,
                spatial_net,
                num_categories,
                start_frame,
                duration)

        avg_spatial_pred = np.mean(spatial_prediction, axis=1)
        avg_temporal_pred = np.mean(temporal_prediction, axis=1)
        avg_total_pred=avg_spatial_pred+avg_temporal_pred
        result_list.append(avg_total_pred)

        pred_index = np.argmax(avg_total_pred)
        print("Sample %d/%d: GT: %d, Prediction: %d" % (line_id, len(val_list), input_video_label, pred_index))

        if pred_index == input_video_label:
            match_count += 1
        line_id += 1

    print(match_count)
    print(len(val_list))
    print("Accuracy is %4.4f" % (float(match_count)/len(val_list)))
    np.save("ucf101_s1_combined_resnet152_tsn_withoutSoft_w1-1.npy", np.array(result_list))

if __name__ == "__main__":
    main()




    # # spatial net prediction
    # class_list = os.listdir(data_dir)
    # class_list.sort()
    # print(class_list)

    # class_index = 0
    # match_count = 0
    # total_clip = 1
    # result_list = []

    # for each_class in class_list:
    #     class_path = os.path.join(data_dir, each_class)

    #     clip_list = os.listdir(class_path)
    #     clip_list.sort()

    #     for each_clip in clip_list:
            # clip_path = os.path.join(class_path, each_clip)
            # spatial_prediction = VideoSpatialPrediction(
            #         clip_path,
            #         spatial_net,
            #         num_categories,
            #         start_frame)

            # avg_spatial_pred_fc8 = np.mean(spatial_prediction, axis=1)
            # # print(avg_spatial_pred_fc8.shape)
            # result_list.append(avg_spatial_pred_fc8)
            # # avg_spatial_pred = softmax(avg_spatial_pred_fc8)

            # pred_index = np.argmax(avg_spatial_pred_fc8)
            # print("GT: %d, Prediction: %d" % (class_index, pred_index))

            # if pred_index == class_index:
            #     match_count += 1
#             total_clip += 1

#         class_index += 1

#     print("Accuracy is %4.4f" % (float(match_count)/total_clip))
#     np.save("ucf101_split1_resnet_rgb.npy", np.array(result_list))

# if __name__ == "__main__":
#     main()
