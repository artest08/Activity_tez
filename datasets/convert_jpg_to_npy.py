#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 17:08:05 2019

@author: esat
"""

import numpy as np 
import cv2
import os
from tqdm import tqdm

datasetName = 'hmdb51'
dataset_dir = './%s_frames' %(datasetName)
target_dir = './%s_numpy' %(datasetName)
folder_list = os.listdir(dataset_dir)


if not os.path.isdir(target_dir):
    os.mkdir(target_dir)
    
for folder in tqdm(folder_list):
    target_folder = os.path.join(target_dir, folder)
    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)
    content_folder = os.path.join(dataset_dir, folder)
    contents = os.listdir(content_folder)
    for content in contents:
        image_location = os.path.join(content_folder,content)
        image = cv2.imread(image_location)
        
        content_name = content.split('.')[0]
        numpy_filename = content_name + '.npy'
        numpy_file_location = os.path.join(target_folder, numpy_filename)
        np.save(numpy_file_location, image)
    
