#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 18:17:46 2019

@author: esat
"""

import json
import os

import re
datasetLocation = 'smtV2_frames'

with open('something-something-v2-test.json') as json_file: # read likedInfoList from db/likedInfoList.json
    dataset = json.load(json_file)
    
with open('something-something-v2-labels.json') as json_file: # read likedInfoList from db/likedInfoList.json
    labels = json.load(json_file)
    
dataset_list = []
for item in dataset:

    videoId = item['id']
    try:
        template = item['template']
        template = template.replace('[','')
        video_class = template.replace(']','')
        classIndex = labels[video_class]
    except:
        classIndex = -1
    videoFile = os.path.join(datasetLocation,videoId)
    frameNum = len(os.listdir(videoFile))
    dataset_list.append('{} {} {}\n'.format(videoId,frameNum,classIndex))
    
open('test', 'w').writelines(dataset_list)