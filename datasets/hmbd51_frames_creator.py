#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:36:41 2019

@author: esat
"""

import os
import shutil

jpegLoc="/media/esat/8234cf14-fc0e-441d-b606-0b7906d5d9c91/tsnCoffe/hmdb51_jpegs_256/jpegs_256"
listJpeg=os.listdir(jpegLoc)
flow_xLoc="/media/esat/8234cf14-fc0e-441d-b606-0b7906d5d9c91/tsnCoffe/hmdb51_tvl1_flow/tvl1_flow/u"
flow_yLoc="/media/esat/8234cf14-fc0e-441d-b606-0b7906d5d9c91/tsnCoffe/hmdb51_tvl1_flow/tvl1_flow/v"
targetLoc="/media/esat/8234cf14-fc0e-441d-b606-0b7906d5d9c91/tsnCoffe/two-stream-pytorch/datasets/hmbd51_frames"

for listItem in listJpeg:
    print("%s adli dosyayi tasiyorum" % (listItem))
    frameList=os.listdir(os.path.join(jpegLoc,listItem))
    frameList.sort()
    isVideoLocExist=os.path.isdir(os.path.join(targetLoc,listItem))
    if not isVideoLocExist:
        os.mkdir(os.path.join(targetLoc,listItem))
    for i,imageName in enumerate(frameList):
        shutil.copy(os.path.join(jpegLoc,listItem,imageName),os.path.join(targetLoc,listItem,"img_%05d.jpg" % (i+1)))
    
    frameList=os.listdir(os.path.join(flow_xLoc,listItem))
    frameList.sort()
    for i,imageName in enumerate(frameList):
        shutil.copy(os.path.join(flow_xLoc,listItem,imageName),os.path.join(targetLoc,listItem,"flow_x_%05d" % (i+1)))
        
    frameList=os.listdir(os.path.join(flow_yLoc,listItem))
    frameList.sort()
    for i,imageName in enumerate(frameList):
        shutil.copy(os.path.join(flow_yLoc,listItem,imageName),os.path.join(targetLoc,listItem,"flow_y_%05d" % (i+1)))
    print("Tasima tamam siradakine geciyorum")