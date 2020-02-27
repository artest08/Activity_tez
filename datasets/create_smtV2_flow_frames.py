#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 20:48:58 2020

@author: esat
"""

import cv2
import os 
from tqdm import tqdm
import argparse
from multiprocessing import Pool

default_loc = os.path.join('..','..','..','optical flow smtV2', '20bn-something-something-v2-flow', 'flow')
parser = argparse.ArgumentParser(description='smtV2 flow formatter')
parser.add_argument('--location', metavar='DIR', default=default_loc,
                    help='path to smtV2 flow frames')


args = parser.parse_args()
directory_list = os.listdir(args.location)
directory_list.sort()
main_target_loc = 'smtV2_frames'
for video_name in tqdm(directory_list):
    flow_image_folder = os.path.join(args.location,video_name)
    flow_image_list = os.listdir(flow_image_folder)
    flow_image_list.sort()
    for image_index, flow_image_name in enumerate(flow_image_list):
        flow_image_loc = os.path.join(flow_image_folder,flow_image_name)
        flow_image = cv2.imread(flow_image_loc)
        flow_x = flow_image[:,:,2]
        flow_y = flow_image[:,:,1]
        target_loc = os.path.join(main_target_loc, video_name)
        target_flowx_name = "flow_x_%05d.jpg" %(image_index + 1)
        target_flowy_name = "flow_y_%05d.jpg" %(image_index + 1)
        target_flowx_loc = os.path.join(target_loc, target_flowx_name)
        target_flowy_loc = os.path.join(target_loc, target_flowy_name)
        cv2.imwrite(target_flowx_loc, flow_x)
        cv2.imwrite(target_flowy_loc, flow_y)

