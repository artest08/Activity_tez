#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 01:25:39 2020

@author: esat
"""

import os
import time
import argparse
import shutil
import numpy as np
import sys

from PIL import Image


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter

from torch.optim import lr_scheduler
from tqdm import tqdm

import video_transforms
import models
import datasets
import swats

from opt.AdamW import AdamW
from weights.model_path import rgb_3d_model_path_selection

# from einops import rearrange
# from einops.layers.torch import Rearrange


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition')
#parser.add_argument('--data', metavar='DIR', default='./datasets/ucf101_frames',
#                    help='path to dataset')
parser.add_argument('--settings', metavar='DIR', default='./datasets/settings',
                    help='path to datset setting files')
#parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb',
#                    choices=["rgb", "flow"],
#                    help='modality: rgb | flow')
parser.add_argument('--dataset', '-d', default='hmdb51',
                    choices=["ucf101", "hmdb51", "smtV2"],
                    help='dataset: ucf101 | hmdb51')
parser.add_argument('--arch', '-a', metavar='ARCH', default='rgb_r2plus1d_32f_34_deep',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: rgb_vgg16)')

parser.add_argument('-s', '--split', default=4, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--iter-size', default=1, type=int,
                    metavar='I', help='iter size as in Caffe to reduce memory usage (default: 5)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[15], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', default=200, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save-freq', default=1, type=int,
                    metavar='N', help='save frequency (default: 25)')
parser.add_argument('--num-seg', default=1, type=int,
                    metavar='N', help='Number of segments for temporal LSTM (default: 16)')
#parser.add_argument('--resume', default='./dene4', type=str, metavar='PATH',
#                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

args = parser.parse_args()

def build_model():
    modelLocation="./checkpoint/"+args.dataset+"_"+'_'.join(args.arch.split('_')[:-1])+"_split"+str(args.split)
    modality=args.arch.split('_')[0]
    if modality == "rgb":
        model_path = rgb_3d_model_path_selection(args.arch)
            
        
    elif modality == "pose":
        model_path=''
        
    elif modality == "flow":
        model_path=''
        if "3D" in args.arch:
            if 'I3D' in args.arch:
                 model_path='./weights/flow_imagenet.pth'   
    elif modality == "both":
        model_path='' 
        
    if args.dataset=='ucf101':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=101,length=args.num_seg)
    elif args.dataset=='hmdb51':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=51, length=args.num_seg)

    if torch.cuda.device_count() > 1:
        model=torch.nn.DataParallel(model)    
    model = model.cuda()
    
    return model
class Denormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, video):
        shape = (-1,) + (1,) * (video.dim() - 1)

        mean = torch.as_tensor(self.mean, device=video.device).reshape(shape)
        std = torch.as_tensor(self.std, device=video.device).reshape(shape)

        return (video * std) + mean

modality=args.arch.split('_')[0]

class TotalVariationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        loss = 0.

        loss += (inputs[:, :, :-1, :, :] - inputs[:, :, 1:, :, :]).abs().sum()
        loss += (inputs[:, :, :, :-1, :] - inputs[:, :, :, 1:, :]).abs().sum()
        loss += (inputs[:, :, :, :, :-1] - inputs[:, :, :, :, 1:]).abs().sum()

        return loss


if '3D' in args.arch:
    if 'I3D' in args.arch or 'MFNET3D' in args.arch:
        if '112' in args.arch:
            scale = 0.5
        else:
            scale = 1
    else:
        scale = 0.5
elif 'r2plus1d' in args.arch:
    scale = 0.5
else:
    scale = 1
    
scale = 1.2
print('scale: %.1f' %(scale))

input_size = int(224 * scale)
width = int(340 * scale)
height = int(256 * scale)
if "3D" in args.arch or 'r2plus1d' in args.arch or "rep_flow" in args.arch:
    if '64f' in args.arch:
        length=64
    elif '32f' in args.arch:
        length=32
    else:
        length=16
elif "tsm" in args.arch:
    length=64
else:
    length = 1
print('length %d' %(length))
# Data transforming
if modality == "rgb":
    is_color = True
    scale_ratios = [1.0, 0.875, 0.75, 0.66]
    if 'I3D' in args.arch:
        if 'resnet' in args.arch:
            clip_mean = [0.45, 0.45, 0.45] * args.num_seg * length
            clip_std = [0.225, 0.225, 0.225] * args.num_seg * length
        else:
            clip_mean = [0.5, 0.5, 0.5] * args.num_seg * length
            clip_std = [0.5, 0.5, 0.5] * args.num_seg * length
        #clip_std = [0.25, 0.25, 0.25] * args.num_seg * length
    elif 'MFNET3D' in args.arch:
        clip_mean = [0.48627451, 0.45882353, 0.40784314] * args.num_seg * length
        clip_std = [0.234, 0.234, 0.234]  * args.num_seg * length
    elif "3D" in args.arch:
        clip_mean = [114.7748, 107.7354, 99.4750] * args.num_seg * length
        clip_std = [1, 1, 1] * args.num_seg * length
    elif "r2plus1d" in args.arch:
        clip_mean = [0.43216, 0.394666, 0.37645] * args.num_seg * length
        clip_std = [0.22803, 0.22145, 0.216989] * args.num_seg * length
    elif "rep_flow" in args.arch:
        clip_mean = [0.5, 0.5, 0.5] * args.num_seg * length
        clip_std = [-0.5, -0.5, -0.5] * args.num_seg * length       
    else:
        clip_mean = [0.485, 0.456, 0.406] * args.num_seg * length
        clip_std = [0.229, 0.224, 0.225] * args.num_seg * length


dataset='./datasets/hmdb51_frames'
normalize = video_transforms.Normalize(mean=clip_mean,
                                       std=clip_std)
denormalize = video_transforms.DeNormalize(mean=clip_mean,
                                       std=clip_std)

if "3D" in args.arch and not ('I3D' in args.arch or 'MFNET3D' in args.arch):
    train_transform = video_transforms.Compose([
            video_transforms.MultiScaleCrop((input_size, input_size), scale_ratios),
            video_transforms.RandomHorizontalFlip(),
            video_transforms.ToTensor2(),
            normalize,
        ])

    val_transform = video_transforms.Compose([
            video_transforms.CenterCrop((input_size)),
            video_transforms.ToTensor2(),
            normalize,
        ])
else:
    train_transform = video_transforms.Compose([
            video_transforms.MultiScaleCrop((input_size, input_size), scale_ratios),
            video_transforms.RandomHorizontalFlip(),
            video_transforms.ToTensor(),
            normalize,
        ])

    val_transform = video_transforms.Compose([
            video_transforms.CenterCrop((input_size)),
            video_transforms.ToTensor(),
            normalize,
        ])

# data loading
train_setting_file = "train_%s_split%d.txt" % (modality, args.split)
train_split_file = os.path.join(args.settings, args.dataset, train_setting_file)
val_setting_file = "val_%s_split%d.txt" % (modality, args.split)
val_split_file = os.path.join(args.settings, args.dataset, val_setting_file)

val_dataset = datasets.__dict__[args.dataset](root=dataset,
                                              source=val_split_file,
                                              phase="val",
                                              modality=modality,
                                              is_color=is_color,
                                              new_length=length,
                                              new_width=width,
                                              new_height=height,
                                              video_transform=val_transform,
                                              num_segments=args.num_seg)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for i, (inputs, targets) in enumerate(val_loader):
    video = inputs
    

# Put video data into grapg leaf node with grads and on device
video = torch.tensor(video, requires_grad=True, device=device)
video=video.view(-1,length,3,input_size,input_size).transpose(1,2)


variation = TotalVariationLoss()

#denormalize = Denormalize(mean=mean, std=std)


progress = tqdm(range(args.epochs))
model = build_model()
model.eval()

for params in model.parameters():
    params.requires_grad = False
for epoch in progress:
    loss = 0.

    acts = model(video)

    # Which channel to maximize normed activations in layer i
    # Channel 6 in layer2 activates on moving eye-like visuals
    channels = [6]
    channels = torch.tensor(channels, device=device, dtype=torch.int64)
    gamma = torch.tensor(1e-7, device=device, dtype=torch.float)

    for act, chn in zip(acts, channels):
        #loss += act.norm()
        loss += act[chn, :, :, :].norm()
        # Instead of maximizing all channels, another option is
        # to maximize specific channel activations; see c above:
        #
        # loss += w * act[:, c, :, :, :].norm()

    # Minimize the total variation regularization term
    tv = -1 * variation(video) * gamma
    loss += tv
    video.retain_grad() 
    loss.backward()

    # Normalize the gradients
    grad = video.grad.data
    grad /= grad.std() + 1e-12

    video.data += args.lr * grad

    # Force video to [0, 1]; note: we are in normalized space
    for i in range(video.size(1)):
        cmin = (0. - clip_mean[i]) / clip_std[i]
        cmax = (1. - clip_mean[i]) / clip_std[i]
        video.data[0, i].clamp_(cmin, cmax)

    video.grad.data.zero_()

    progress.set_postfix({"loss": loss.item(), "tv": tv.item()})

# Once we have our dream, denormalize it,
# and turn it into sequence of PIL images.

video = video.squeeze(0).transpose(0,1)
video = video.view(length * 3,input_size,input_size)
video = video.cpu()
video = denormalize(video)
video = video.view(3, length, input_size,input_size)
video = video.permute([1,2,3,0])
video.clamp_(0, 1)
video = video.data.cpu().numpy()


assert video.shape[0] == 32
assert video.shape[3] == 3

assert video.dtype == np.float32
assert (video >= 0).all()
assert (video <= 1).all()

video = (video * 255).astype(np.uint8)

images = [Image.fromarray(v, mode="RGB") for v in video]

images[0].save('deneme', format="GIF", append_images=images[1:],
                save_all=True, duration=(1000 / 30), loop=0)

print("💤 Done", file=sys.stderr)

