#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:58:46 2018

@author: esat
"""

import os
import time
import argparse
import shutil
import numpy as np

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import video_transforms
import models
import datasets


def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    print("Building model ... ")
    
    model_path = 'checkpoints-rgb/model_best.pth.tar'
    data_dir = "datasets/ucf101_frames"
    model = build_model(model_path)
    print("Model %s is loaded. " % (args.arch))


    # Data transforming
    if args.modality == "rgb":
        is_color = True
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        clip_mean = [0.485, 0.456, 0.406] * args.new_length
        clip_std = [0.229, 0.224, 0.225] * args.new_length
    elif args.modality == "flow":
        is_color = False
        scale_ratios = [1.0, 0.875, 0.75]
        clip_mean = [0.5, 0.5] * args.new_length
        clip_std = [0.226, 0.226] * args.new_length
    else:
        print("No such modality. Only rgb and flow supported.")

    normalize = video_transforms.Normalize(mean=clip_mean,
                                           std=clip_std)
    train_transform = video_transforms.Compose([
            # video_transforms.Scale((256)),
            video_transforms.MultiScaleCrop((224, 224), scale_ratios),
            video_transforms.RandomHorizontalFlip(),
            video_transforms.ToTensor(),
            normalize,
        ])


    # data loading
    train_setting_file = "train_%s_split%d.txt" % (args.modality, args.split)
    train_split_file = os.path.join(args.settings, args.dataset, train_setting_file)

    train_dataset = datasets.__dict__[args.dataset](root=args.data,
                                                    source=train_split_file,
                                                    phase="train",
                                                    modality=args.modality,
                                                    is_color=is_color,
                                                    new_length=args.new_length,
                                                    new_width=args.new_width,
                                                    new_height=args.new_height,
                                                    video_transform=train_transform,
                                                    num_segments=25)

    print('{} train samples are found'.format(len(train_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=5, shuffle=False,
        num_workers=8, pin_memory=True)
    
    for i, (input, target) in enumerate(train_loader):
        print()



def build_model(model_path):
    model_start_time = time.time()
    params = torch.load(model_path)
    model = models.rgb_resnet152(pretrained=False, num_classes=101)
    model.load_state_dict(params['state_dict'])
    model.cuda()
    model.eval()
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print("Action recognition model is loaded in %4.4f seconds." % (model_time))
    return model



if __name__ == '__main__':
    main()
