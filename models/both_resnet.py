#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:21:48 2019

@author: esat
"""

import torch.nn as nn
import torch
import math
from collections import OrderedDict
from .BERT.bert import BERT, BERT2, BERT3, BERT4, BERT5
import torch.utils.model_zoo as model_zoo
from .flow_resnet import flow_resnet18
from .rgb_resnet import rgb_resnet18

__all__ = ['both_resnet18_bert10']




class both_resnet18_bert10(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(both_resnet18_bert10, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        self.features1_flow=nn.Sequential(*list(flow_resnet18(pretrained=True,input_frame=2).children())[:-5])
        self.features2_flow=nn.Sequential(*list(flow_resnet18(pretrained=True,input_frame=2).children())[-5:-3])
     
        self.features1_rgb=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-5])
        self.features2_rgb=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[-5:-3])     
        
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT5(512,self.length*2, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(512, num_classes)
            
        for param in self.features1_flow.parameters():
            param.requires_grad = True
        for param in self.features2_flow.parameters():
            param.requires_grad = True
            
        for param in self.features1_rgb.parameters():
            param.requires_grad = False
        for param in self.features2_flow.parameters():
            param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x_flow = x[:,-2:,...]
        x_rgb = x[:,:-2,...]
        x_flow = self.features1_flow(x_flow)
        x_flow = self.features2_flow(x_flow)
        x_flow = self.avgpool(x_flow)
        x_flow = x_flow.view(x_flow.size(0), -1)
        x_flow = x_flow.view(-1,self.length,512)
        
        x_rgb = self.features1_rgb(x_rgb)
        x_rgb = self.features2_rgb(x_rgb)
        x_rgb = self.avgpool(x_rgb)
        x_rgb = x_rgb.view(x_rgb.size(0), -1)
        x_rgb = x_rgb.view(-1,self.length,512)
        
        x = torch.cat((x_rgb,x_flow),1)
        input_vectors=x
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample