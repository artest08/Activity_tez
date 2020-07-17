#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:55:06 2020

@author: esat
"""

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
import cv2
import numpy as np
import time
from numpy import save
from tqdm import tqdm
import random as rnd
import copy 

np.random.seed(19680801)
colorArray = np.random.rand(25, 3) * 255

sys.path.append('/home/esat/openpose/build/python')
from openpose import pyopenpose as op


scale = 1
width = 340 * scale
height = 256 * scale
interpolation = cv2.INTER_LINEAR


params = dict()
params["model_folder"] = "/home/esat/openpose/models"

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum = op.Datum()



exitBool = False
dataset_name = 'window'
selectedMainFolder = os.path.join('%s_frames' %(dataset_name))
writefolder = os.path.join('pose_information2',dataset_name)
if not os.path.isdir(writefolder):
    os.mkdir(writefolder)
selectedVideoList = os.listdir(selectedMainFolder)
totalSample = len(selectedVideoList)
max_people_count = 10


    
for i in tqdm(range(totalSample)):
    folder = selectedVideoList[i]
    #print('%s is starting, %d/%d' %(folder,i,totalSample))
    selectedFolderName = os.path.join(selectedMainFolder,folder)
    writeFolderName = os.path.join(writefolder,folder)
    if not os.path.isdir(writeFolderName):
        os.mkdir(writeFolderName)
        
    old_pose_list = []
    IDlist_history = []
    for i in range(max_people_count):
        IDlist_history.append([])
    fileList = os.listdir(selectedFolderName)
    filteredFileList = [file for file in fileList if 'img' in file]
    filteredFileList.sort()

    frame_pose_info_list = []
    for file in filteredFileList:
        fileLoc = os.path.join(selectedFolderName, file)
        frame_unresized = cv2.imread(fileLoc)
        frame = cv2.resize(frame_unresized, (width, height), interpolation)
        
        
        imageToProcess = frame
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])
        
        file_replaced = file.replace('img', 'pose1')
        fileLocWrite = os.path.join(selectedFolderName, file_replaced)
        
        
        #cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData - frame)
        
        #cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData)
        
        #cv2.imshow('video',frame)
        cv2.imwrite(fileLocWrite, datum.cvOutputData - frame)
        
        key=cv2.waitKey(1)
        if key==27:
            exitBool=True
            break
        if key==ord('p'):
            while True:
                key=cv2.waitKey(1)
                if key==ord('p'):
                    break

cv2.destroyAllWindows()    
