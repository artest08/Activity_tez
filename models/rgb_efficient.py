#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:36:52 2019

@author: esat
"""

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import sys
from time import time
from models.poseNet import openPoseL2Part
from models.convGRU import ConvGRU
from .BERT.bert import BERT, BERT2, BERT3, BERT4, BERT5
from .BERT.embedding import BERTEmbedding
import torch
import numpy as np

__all__ = ['rgb_efficientB1_bert3','rgb_efficientB1_bert2']


from .efficientnet_pytorch import EfficientNet


class rgb_efficientB1_bert3(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_efficientB1_bert3, self).__init__()
        self.hidden_size=1280
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        self.loadedPretrainedModel = EfficientNet.from_pretrained('efficientnet-b1', num_classes=51)
        #self.features=nn.Sequential(*list(self.loadedPretrainedModel.children())[:-2])
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT5(self.hidden_size,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
        for param in self.loadedPretrainedModel.parameters():
            param.requires_grad = False
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.loadedPretrainedModel.extract_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = x.view(-1,self.length,self.hidden_size)
        input_vectors=x
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
    
class rgb_efficientB1_bert2(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_efficientB1_bert2, self).__init__()
        self.hidden_size=1280
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        self.loadedPretrainedModel = EfficientNet.from_pretrained('efficientnet-b1', num_classes=51)
        #self.features=nn.Sequential(*list(self.loadedPretrainedModel.children())[:-2])
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT5(self.hidden_size,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
        for param in self.loadedPretrainedModel.parameters():
            param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.loadedPretrainedModel.extract_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = x.view(-1,self.length,self.hidden_size)
        input_vectors=x
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample