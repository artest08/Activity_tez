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
from multiprocessing import Pool

default_loc = os.path.join('..','..','..','20bn-something-something-v2')
parser = argparse.ArgumentParser(description='Video to image converter')
parser.add_argument('--location', metavar='DIR', default=default_loc,
                    help='path to datset setting files')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

# def convert(videoFileName):
#     videoFile = os.path.join(videoFolderLocation,videoFileName)
#     cap=cv2.VideoCapture(videoFile)     
#     frameNum = 1    
#     imageFolderName = os.path.join(imageDirectory,videoFileName[:-5])   
#     if not os.path.isdir(imageFolderName):
#         os.mkdir(imageFolderName)
#     while True:
#         ret,frame=cap.read()
#         if ret==False:
#             break
#         imageFileName = os.path.join(imageFolderName,'img_%05d.jpg' %(frameNum))
#         cv2.imwrite(imageFileName, frame)
#         frameNum = frameNum+1
#     cap.release()
#     cv2.destroyAllWindows() 
#     counter = counter+1
#     print('{} done {}/{} completed'.format(videoFileName,counter,total),flush=True)
def main():
    global imageDirectory, videoFolderLocation
    args = parser.parse_args()
    videoFolderLocation = args.location
    imageDirectory = './window_frames'
    if not os.path.isdir(imageDirectory):
        os.mkdir(imageDirectory)
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
# def main():
#     global imageDirectory, videoFolderLocation, counter, total
#     args = parser.parse_args()
#     videoFolderLocation = args.location
#     imageDirectory = './window_frames'
#     if not os.path.isdir(imageDirectory):
#         os.mkdir(imageDirectory)
#     files = os.listdir(videoFolderLocation)
#     total = len(files)
#     counter = 0
#     with Pool(processes=args.workers) as pool:
#         pool.map(convert, files)
        
if __name__ == '__main__':
    main()
    
     