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

width = 340
height = 256
interpolation = cv2.INTER_LINEAR






class pose:
    def __init__(self, poseInfoArray, poseScore, frame, grayscale_image, ID):
        self.ID = ID
        self.track_len = 5
        self.poseInfoArray = poseInfoArray
        self.poseInfoList = poseInfoArray[:,:2].tolist()
        self.poseScore = poseScore
        self.poseInfoListInt = poseInfoArray[:,:2].astype(int).tolist()
        self.problematic_joint_list = self.__check_problematic_joint()
        self.track_color = (0,255,0)
        self.estimated_track_color = (0,0,255)
        self.available_joint = 25 - np.sum(self.problematic_joint_list)
        self.search_distance_klt = 5
        
        self.feature_params = dict( maxCorners = self.available_joint * 3,
                               qualityLevel = 0.01,
                               minDistance = 3,
                               blockSize = 3 ) 
        
        self.lk_params = dict( winSize  = (5, 5),
                  maxLevel = 32,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01))
        
        self.frame = frame
        self.grayscale_image = grayscale_image
        self.klt_points = self.__determine_KLT_point()
        self.corresponding_joint_to_klt_list = self.__find_corresponding_joint_to_klt()
        self.history =self. __create_joint_history()
        self.estimated_klt = None
        self.goodness = None
        
    def update_ID(self, ID):
        self.ID = ID
    def drawPoint(self, frame):
        for jointID in range(25):
            if self.problematic_joint_list[jointID] == 1:
                continue
            joint = self.poseInfoListInt[jointID]
            color = tuple(colorArray[jointID])
            joint_location = tuple(joint)
            cv2.circle(self.frame, joint_location, 1, color , -1)
    def printID_to_frame(self, frame):
        available_joint_ID = 0
        while self.problematic_joint_list[available_joint_ID] != 0:
            available_joint_ID += 1
        cv2.putText(frame, 
        str(self.ID), 
        tuple(self.poseInfoListInt[available_joint_ID]), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1e-3 * 720, 
        (0,180,0), 2)
    def average_joint_coordinates(self):
        pose_info_array = np.array(self.poseInfoList)
        problematic_joint_array = np.array(self.problematic_joint_list)
        selected_joints = pose_info_array[problematic_joint_array == 0]
        mean_joints = np.mean(selected_joints, 0)
        return mean_joints
        
    def __create_joint_history(self):
        history = []
        history.append(self.poseInfoList)
        return history
    def __check_problematic_joint(self):
        problematic_joint_list = []
        for jointID in range(25):
            if self.poseInfoList[jointID] == [0,0]:
                '''  1 defines the joint is problematic'''
                problematic_joint_list.append(1)
            else:
                problematic_joint_list.append(0)
        return problematic_joint_list
    def draw_estimated_klt(self,frame):
        if self.estimated_klt is not None:
            for good, (x, y) in zip(self.goodness, np.float32(self.estimated_klt).reshape(-1, 2)):
                if good:
                    cv2.circle(frame, (x, y), 2, self.estimated_track_color , -1)
    def __determine_KLT_point(self):
        mask = np.zeros([height,width])
        mask = mask.astype(np.uint8)
        for jointID in range(25):
            if self.problematic_joint_list[jointID] == 1:
                continue
            joint = self.poseInfoListInt[jointID]
            joint_location = tuple(joint)
            cv2.circle(mask, joint_location, self.search_distance_klt , 255, -1)
        klt_points = cv2.goodFeaturesToTrack(
            self.grayscale_image, mask = mask, **self.feature_params)
        if klt_points is not None:
            for x, y in np.float32(klt_points).reshape(-1, 2):
                cv2.circle(self.frame, (x, y), 2, self.track_color , -1)
            return klt_points
        else:
            return []
    def __find_corresponding_joint_to_klt(self):
        corresponding_joint_to_klt_list = []
        for klt_point_index in range(len(self.klt_points)):
            corresponding_joint_to_klt = np.argmin(
                np.sqrt(np.sum(np.square(self.klt_points[klt_point_index] - self.poseInfoList),1)))
            corresponding_joint_to_klt_list.append(corresponding_joint_to_klt)
        return corresponding_joint_to_klt_list
    
    
    def estimate_next_frame_klt_locations(self,next_grayscale_image):
        if len(self.klt_points) > 0:
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(
                self.grayscale_image, next_grayscale_image, self.klt_points, None, **self.lk_params)
            p0r, _st, _err = cv2.calcOpticalFlowPyrLK(
                next_grayscale_image, self.grayscale_image, p1, None, **self.lk_params)
            d = abs(self.klt_points-p0r).reshape(-1, 2).max(-1)
            goodness = d < 1
            self.estimated_klt = p0r
            self.goodness = goodness
        else:
            self.estimated_klt = []
            self.goodness = [] 


params = dict()
params["model_folder"] = "/home/esat/openpose/models"

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum = op.Datum()



exitBool = False
dataset_name = 'hmdb51'
selectedMainFolder = os.path.join('..','datasets','%s_frames' %(dataset_name))
writefolder = os.path.join('..','datasets','pose_information2',dataset_name)
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
        
        
        
        # cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData - frame)
        
     
        # cv2.imshow('video', datum.cvOutputData)
        # if not datum.poseKeypoints.size == 1:
        #     pose_array = datum.poseKeypoints[:,:,:-1]/ np.array([frame.shape[0], frame.shape[1]])
        # else:
        #     pose_array = np.zeros([1,25,2])
        

        
        
        IDlist=[pose_instance.ID for pose_instance in old_pose_list]  
        for old_pose_index, old_pose in enumerate(old_pose_list):
            if old_pose_index < max_people_count: 
                IDlist_history[old_pose_index].append(old_pose.ID)
        
        pose_list = []
        grayscale_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if not datum.poseKeypoints.size == 1:
            pose_count =  datum.poseKeypoints.shape[0]
            for pose_id in range(pose_count):
                ID=rnd.randint(0,99)
                while ID in IDlist:
                    ID=rnd.randint(0,99)
                    
                pose_instance = pose(datum.poseKeypoints[pose_id],
                                     datum.poseScores[pose_id], frame, grayscale_image, ID)
                pose_list.append(pose_instance)
                IDlist.append(ID)
                
        deleted_old_pose_list = old_pose_list.copy()
        if len(old_pose_list) == 0:
            old_pose_list = pose_list              

        elif  len(pose_list) == 0:
            old_pose_list = old_pose_list
           
        else:
            match_matrix = np.zeros([len(old_pose_list),len(pose_list)])
            count_vector = np.zeros([len(old_pose_list)])
            for i,old_pose in enumerate(old_pose_list):
                old_pose.estimate_next_frame_klt_locations(grayscale_image)
                old_pose.draw_estimated_klt(frame)
                for j,new_pose in enumerate(pose_list):
                    for k in range(len(old_pose.estimated_klt)):
                        if old_pose.goodness[k] == True:
                            count_vector[i] += 1
                            distance = np.sqrt(np.sum(np.square(old_pose.estimated_klt[k]
                                                                - new_pose.poseInfoList),1))
                            min_distance_value = np.min(distance)
                            if min_distance_value < 2 * old_pose.search_distance_klt * np.sqrt(2):
                                match_matrix[i][j] += 1
            count_vector = count_vector/ len(pose_list)     
            klt_match_indexes = np.argmax(match_matrix,0)
            is_updated_boolean = []
            for pose_index,pose_instance in enumerate(pose_list):
                matching_previous_pose_index = klt_match_indexes[pose_index]
                matching_previous_pose = old_pose_list[matching_previous_pose_index]
                if count_vector[matching_previous_pose_index] != 0:
                    matching_ratio = (match_matrix[matching_previous_pose_index, pose_index]
                    / count_vector[matching_previous_pose_index])
                else:
                    matching_ratio = 0
                pose_index_check = np.argmax(match_matrix[matching_previous_pose_index])
                if matching_ratio > 0.10 and pose_index==pose_index_check:
                    matching_previous_pose_ID = matching_previous_pose.ID
                    pose_instance.update_ID(matching_previous_pose_ID)
                    is_updated_boolean.append(True)
                    deleted_old_pose_list_matching_index = deleted_old_pose_list.index(matching_previous_pose)
                    del deleted_old_pose_list[deleted_old_pose_list_matching_index]
                else:
                    is_updated_boolean.append(False)

                  

            for pose_index,pose_instance in enumerate(pose_list):   
                if is_updated_boolean[pose_index] == True:
                    continue
                if len(deleted_old_pose_list) == 0:
                    continue
                min_distance = None
                min_pose = None
                min_index = None
                for old_pose_index,old_pose in enumerate(deleted_old_pose_list):
                    distances_to_be_processed = (
                        pose_instance.average_joint_coordinates() - old_pose.average_joint_coordinates())
                    distance = np.sum(np.square(distances_to_be_processed))
                    if min_distance == None:
                        min_distance_check = None
                        for pose_index_check,pose_instance_check in enumerate(pose_list):   
                            if is_updated_boolean[pose_index_check] == True:
                                continue
                            distances_to_be_processed_check = (
                            pose_instance_check.average_joint_coordinates() - old_pose.average_joint_coordinates())
                            distance_check = np.sum(np.square(distances_to_be_processed_check))
                            if min_distance_check == None:
                                min_distance_check = distance_check
                            elif distance_check < min_distance_check:
                                min_distance_check = distance_check           
                        if not min_distance_check < distance:
                            min_distance = distance
                            min_pose = copy.copy(old_pose)
                            min_index = old_pose_index
                    elif distance < min_distance:
                        min_distance_check = None
                        for pose_index_check,pose_instance_check in enumerate(pose_list):   
                            if is_updated_boolean[pose_index_check] == True:
                                continue
                            distances_to_be_processed_check = (
                            pose_instance_check.average_joint_coordinates() - old_pose.average_joint_coordinates())
                            distance_check = np.sum(np.square(distances_to_be_processed_check))
                            if min_distance_check == None:
                                min_distance_check = distance_check
                            elif distance_check < min_distance_check:
                                min_distance_check = distance_check           
                        if not min_distance_check < distance:
                            min_distance = distance
                            min_pose = copy.copy(old_pose)
                            min_index = old_pose_index
                if min_pose is not None:
                    matching_previous_pose_ID = min_pose.ID
                    pose_instance.update_ID(matching_previous_pose_ID)
                    is_updated_boolean[pose_index] = True
                    del deleted_old_pose_list[min_index]
                

            old_pose_list = copy.copy(pose_list)         
            old_pose_list += deleted_old_pose_list
                        
        for pose_index,pose_instance in enumerate(pose_list):  
            pose_instance.drawPoint(frame)
            pose_instance.printID_to_frame(frame)

                
                 
            
        frame_pose_info_list.append(pose_list)
        cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData)
        
 
        cv2.imshow('video',frame)
        #save(fileLocWrite, pose_array)
        #cv2.imwrite(fileLocWrite, datum.cvOutputData - frame)
        
        key=cv2.waitKey(1)
        if key==27:
            exitBool=True
            break
        if key==ord('p'):
            while True:
                key=cv2.waitKey(1)
                if key==ord('p'):
                    break
    
    ID_frequency = {}
    for frame_pose_info in frame_pose_info_list:
        for pose_instance in frame_pose_info:
            ID = pose_instance.ID
            if ID in ID_frequency.keys():
                ID_value = ID_frequency[ID]
                ID_frequency.update({ID: ID_value + 1})
            else:
                ID_frequency.update({ID: 1})
    sorted_list = sorted([(v,k) for k,v in ID_frequency.items()])
    sorted_list = sorted_list[::-1]
    sorted_IDs = [k for v,k in sorted_list]
    selected_sorted_IDs = sorted_IDs[:max_people_count]
    for frame_index in range(len(frame_pose_info_list)):
        pose_info = np.zeros([10,25,2])
        frame_pose_info = frame_pose_info_list[frame_index]
        for pose_instance in frame_pose_info:
            pose_ID = pose_instance.ID
            if pose_ID in selected_sorted_IDs:
                ID_index = selected_sorted_IDs.index(pose_ID)
                pose_info[ID_index, :, :] = np.array(pose_instance.poseInfoList)
        # fileLocWrite = os.path.join(writeFolderName, 'pose_%05d.npy' %(frame_index+1))
        # np.save(fileLocWrite, pose_info)
#    print(ID_frequency)
    if exitBool:
        break


cv2.destroyAllWindows()    
