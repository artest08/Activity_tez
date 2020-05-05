#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 20:27:26 2020

@author: esat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


import os
import sys
from collections import OrderedDict


from .BERT.bert import BERT, BERT2, BERT3, BERT4, BERT5, BERT6
from .SlowFast.slowfast_connector import slowfast_50


__all__ = ['rgb_slowfast64f_50', 'rgb_slowfast64f_50_bert10']

class rgb_slowfast64f_50(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_slowfast64f_50, self).__init__()
        self.model = slowfast_50(modelPath)
        self.num_classes=num_classes
        self.model.head.dropout = nn.Dropout(0.8)
        self.fc_action = nn.Linear(2304, num_classes)
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        self.model.head.projection = self.fc_action
    def forward(self, x):
        fast_input = x[:, :, ::2, :, :]
        slow_input = x[:, :, ::8, :, :]
        x = self.model.forward([slow_input, fast_input])
        x = x.view(-1, self.num_classes)
        #x = self.model.forward([fast_input, slow_input])
        return x
    
    def mars_forward(self, x):
        fast_input = x[:, :, ::2, :, :]
        slow_input = x[:, :, ::8, :, :]
        x = self.model.forward_feature([slow_input, fast_input])
        slow_feature = self.avgpool(x[0])
        fast_feature = self.avgpool(x[1])
        slow_feature = slow_feature.view(-1, 2048)
        fast_feature = fast_feature.view(-1, 256)
        features = torch.cat([slow_feature, fast_feature], 1)
        return features
    
class rgb_slowfast64f_50_bert10(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_slowfast64f_50_bert10, self).__init__()
        self.hidden_size=256
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        self.avgpool_fast = nn.AvgPool3d((1, 7, 7), stride=1)
        self.avgpool_slow = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.model = slowfast_50(modelPath)   
        self.model.head.dropout = nn.Dropout(0.8)
        
        self.bert = BERT5(self.hidden_size, 32 , hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        
        self.fc_action = nn.Linear(2304, num_classes)
            
        for param in self.model.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        fast_input = x[:, :, ::2, :, :]
        slow_input = x[:, :, ::8, :, :]
        x = self.model.forward_feature([slow_input, fast_input])
        slow_feature = x[0]
        fast_feature = x[1]
        slow_feature = self.avgpool_slow(slow_feature)
        fast_feature = self.avgpool_fast(fast_feature)
        slow_feature = slow_feature.view(-1, 2048)
        
        fast_feature = fast_feature.view(fast_feature.size(0), self.hidden_size, 32)
        fast_feature = fast_feature.transpose(1,2)
        input_vectors = fast_feature
        output , maskSample = self.bert(fast_feature)
        fast_feature_out = output[:,0,:]
        sequenceOut=output[:,1:,:]
        classificationOut = torch.cat([slow_feature, fast_feature_out], 1)
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
        