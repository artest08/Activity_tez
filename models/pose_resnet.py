#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:10:14 2019

@author: esat
"""

import torch.nn as nn
import torch
import math
from collections import OrderedDict
from .BERT.bert import BERT, BERT2, BERT3, BERT4, BERT5
import torch.utils.model_zoo as model_zoo
from .rgb_resnet import rgb_resnet18

__all__ = ['pose_resnet18_bert10']



class pose_resnet18_bert10(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(pose_resnet18_bert10, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        

        self.features1=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-5])
        self.features2=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[-5:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT5(512,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(512, num_classes)
            
        for param in self.features1.parameters():
            param.requires_grad = True
        for param in self.features2.parameters():
            param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = x.view(-1,self.length,512)
        input_vectors=x
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample