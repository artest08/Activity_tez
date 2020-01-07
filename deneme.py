#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:00:25 2019

@author: esat
"""
from models.poseNet import openPoseL2Part,openPose
import video_transforms
import torch
import cv2
import numpy as np
from numpy import linalg as LA

clip_mean = [0.485, 0.456, 0.406] 
clip_std = [0.229, 0.224, 0.225] 

normalize = video_transforms.Normalize(mean=clip_mean,
                                       std=clip_std)

val_transform = video_transforms.Compose([
        # video_transforms.Scale((256)),
        video_transforms.CenterCrop((224)),
        video_transforms.ToTensor(),
        normalize,
    ])

image_transform = video_transforms.Compose([
        # video_transforms.Scale((256)),
        video_transforms.CenterCrop((224))
    ])

newModel=openPoseL2Part()
model=openPose()
if torch.cuda.is_available():
    newModel = newModel.cuda()
    model=model.cuda()

newModel.eval()
model.eval()

image=cv2.imread('datasets/hmdb51_frames/3-er_handstand_aua_handstand_u_cm_np3_ba_med_0/img_00004.jpg')

imageNew=image_transform(image)
clip_input = val_transform(image)
clip_input=torch.unsqueeze(clip_input,0)
clip_input=clip_input.cuda()
with torch.no_grad():
    Mconv7_stage3_L2 = newModel(clip_input)
    Mconv7_stage1_L1, Mconv7_stage3_L2 = model(clip_input)
    
    
Mconv7_stage3_L2 = Mconv7_stage3_L2.cpu().numpy()
Mconv7_stage3_L2 = np.squeeze(Mconv7_stage3_L2,0)

Mconv7_stage1_L1 = Mconv7_stage1_L1.cpu().numpy()
Mconv7_stage1_L1 = np.squeeze(Mconv7_stage1_L1,0)

Mconv7_stage3_L2_reshaped=Mconv7_stage3_L2.reshape([26,-1,28,28])


Mconv7_stage3_L2_normed=LA.norm(Mconv7_stage3_L2_reshaped, axis=1)
mask=np.max(Mconv7_stage3_L2_normed,0)
maskResized=cv2.resize(mask, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
maskResized=np.concatenate([maskResized[:,:,np.newaxis],maskResized[:,:,np.newaxis],maskResized[:,:,np.newaxis]],axis=2)
newMaskResized=(maskResized+imageNew/256)/2
#cv2.imshow('maskx',newMaskResized)
cv2.waitKey()
for i in range(0,26):
    mask1=Mconv7_stage1_L1[i,:,:]
    mask2=np.abs(Mconv7_stage3_L2[2*i,:,:])
    mask3=np.abs(Mconv7_stage3_L2[2*i+1,:,:])
    
#    mask2=Mconv7_stage3_L2[2*i,:,:]
#    mask3=Mconv7_stage3_L2[2*i+1,:,:]
#    
    maskResized1=cv2.resize(mask1, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    maskResizedx=cv2.resize(mask2, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    maskResizedy=cv2.resize(mask3, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    
    maskResized1=np.concatenate([maskResized1[:,:,np.newaxis],maskResized1[:,:,np.newaxis],maskResized1[:,:,np.newaxis]],axis=2)
    newMaskResized1=(maskResized1+imageNew/256)/2
    
    maskResizedx=np.concatenate([maskResizedx[:,:,np.newaxis],maskResizedx[:,:,np.newaxis],maskResizedx[:,:,np.newaxis]],axis=2)
    newMaskResizedx=(maskResizedx+imageNew/256)/2
    
    maskResizedy=np.concatenate([maskResizedy[:,:,np.newaxis],maskResizedy[:,:,np.newaxis],maskResizedy[:,:,np.newaxis]],axis=2)
    newMaskResizedy=(maskResizedy+imageNew/256)/2
    
    cv2.imshow('mask1',newMaskResized1)
    cv2.imshow('maskx',maskResizedx)
    cv2.imshow('masky',maskResizedy)
    cv2.waitKey()
    
cv2.destroyAllWindows()    

