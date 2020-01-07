#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:19:42 2019

@author: esat
"""

import cv2
import os 
from tqdm import tqdm
import argparse

default_loc = os.path.join('..','..','..','20bn-something-something-v2')
parser = argparse.ArgumentParser(description='Video to image converter')
parser.add_argument('--location', metavar='DIR', default=default_loc,
                    help='path to datset setting files')

def main():
    global args
    videoFolderLocation = args.location
    imageDirectory = 'smtV2_frames'
    files = os.listdir(videoFolderLocation)
    for videoFileName in tqdm(files):
        videoFile = os.path.join(videoFolderLocation,videoFileName)
        cap=cv2.VideoCapture(videoFile)     
        frameNum = 1    
        imageFolderName = os.path.join(imageDirectory,videoFileName[:-5])   
        if not os.path.isdir(imageFolderName):
            os.mkdir(imageFolderName)
        while True:
            ret,frame=cap.read()
            if ret==False:
                break
            imageFileName = os.path.join(imageFolderName,'img_%05d.jpg' %(frameNum))
            cv2.imwrite(imageFileName, frame)
            frameNum = frameNum+1
            
        cap.release()
        cv2.destroyAllWindows()    
    
     