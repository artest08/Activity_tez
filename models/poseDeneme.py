#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 00:55:09 2019

@author: esat
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:43:23 2019

@author: esat
"""
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import sys
import cv2
import numpy as np
import time

sys.path.append('/home/esat/openpose/build/python')
from openpose import pyopenpose as op




params = dict()
params["model_folder"] = "/home/esat/openpose/models"

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum = op.Datum()

selectedMainFolder = '/media/esat/6a7c4273-8106-47bc-b992-6760dfcea9a1/tsnCoffe/two-stream-pytorch/datasets/ucf101_frames/'
selectedVideoList = os.listdir(selectedMainFolder)
totalSample = len(selectedVideoList)
for i,folder in enumerate(selectedVideoList):
    
    print('%s is starting, %d/%d' %(folder,i,totalSample))
    selectedFolderName = os.path.join(selectedMainFolder,folder)

    fileList = os.listdir(selectedFolderName)
    filteredFileList = [file for file in fileList if 'img' in file]
    filteredFileList.sort()

    
    for file in filteredFileList:
        fileLoc = os.path.join(selectedFolderName, file)
        frame = cv2.imread(fileLoc)
        
        
        imageToProcess = frame
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])
        
        
        
        cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData - frame)
        
     
        cv2.imshow('video', datum.cvOutputData)
        
        fileLocWrite = os.path.join(selectedFolderName, 'pose1_%s' %(file[4:]))
        cv2.imwrite(fileLocWrite, datum.cvOutputData - frame)
        
        key=cv2.waitKey(1)
        if key==27:
            break
        if key==ord('p'):
            while True:
                key=cv2.waitKey(1)
                if key==ord('p'):
                    break


cv2.destroyAllWindows()    
