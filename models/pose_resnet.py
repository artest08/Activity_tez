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
from .BERT.bert import BERT, BERT2, BERT3, BERT4, BERT5, BERT7
import torch.utils.model_zoo as model_zoo
from .rgb_resnet import rgb_resnet18, rgb_resnet101

__all__ = ['pose_resnet18_bert10', 'poseRaw_bert10', 'poseRaw_bert10X', 'poseRaw_bert10_newExtractedPoses',
           'poseRaw_bert10_twoPeople','poseRaw2_bert7','poseRaw_bert10_twoPeople_withoutAngle'
           ,'pose_resnet101_pooling5']



class pose_resnet101_pooling5(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(pose_resnet101_pooling5, self).__init__()
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        self.avgpool1 = nn.AvgPool2d(28)
        self.avgpool2 = nn.AvgPool2d(14)
        self.avgpool3 = nn.AvgPool2d(7)

        if modelPath=='':
            self.features=nn.Sequential(*list(rgb_resnet101(pretrained=True).children())[:-6])
            self.features1=nn.Sequential(*list(rgb_resnet101(pretrained=True).children())[-6])
            self.features2=nn.Sequential(*list(rgb_resnet101(pretrained=True).children())[-5])
            self.features3=nn.Sequential(*list(rgb_resnet101(pretrained=True).children())[-4])

        for param in self.features.parameters():
            param.requires_grad = True        
        for param in self.features1.parameters():
            param.requires_grad = True
        for param in self.features2.parameters():
            param.requires_grad = True
        for param in self.features3.parameters():
            param.requires_grad = True

        self.fc_action = nn.Linear((512 + 1024 + 2048)*self.length, self.num_classes)
        
        
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
    
    def forward(self, x):
        x = self.features(x)
        features1 = self.features1(x)
        features2 = self.features2(features1)
        features3 = self.features3(features2)
        features1 = self.avgpool1(features1)
        features2 = self.avgpool2(features2)
        features3 = self.avgpool3(features3)
        features1 = features1.view(-1,self.length,512)
        features2 = features2.view(-1,self.length,1024)
        features3 = features3.view(-1,self.length,2048)
        #features1 = self.mapper1(features1)
        #features2 = self.mapper2(features2)
        x = torch.cat((features1,features2,features3),2)
        input_and_output = x
        x=x.view(-1,(512 + 1024 + 2048)*self.length)
        x = self.dp(x)
        x = self.fc_action(x)
        

        return x, input_and_output, input_and_output, input_and_output

class pose_resnet18_bert10(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(pose_resnet18_bert10, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        

        self.features1=nn.Sequential(*list(rgb_resnet18(pretrained=False).children())[:-5])
        self.features2=nn.Sequential(*list(rgb_resnet18(pretrained=False).children())[-5:-3])
        
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
    
class poseRaw_bert10(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(poseRaw_bert10, self).__init__()
        self.hidden_size=100
        self.n_layers=1
        self.attn_heads=2
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.2)
        
    
        self.bert = BERT5(self.hidden_size,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = x.view(-1,self.length,self.hidden_size)
        input_vectors=x
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
    
    
class poseRaw_bert10_newExtractedPoses(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(poseRaw_bert10_newExtractedPoses, self).__init__()
        self.hidden_size=390
        self.n_layers=1
        self.attn_heads=1
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
    
        self.bert = BERT5(self.hidden_size,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = x.view(-1,self.length,self.hidden_size)
        input_vectors=x
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
    
    
class poseRaw_bert10_twoPeople(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(poseRaw_bert10_twoPeople, self).__init__()
        self.hidden_size=156
        self.n_layers=1
        self.attn_heads=1
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
    
        self.bert = BERT5(self.hidden_size,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = x.view(-1,self.length,self.hidden_size)
        input_vectors=x
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
    
class poseRaw_bert10_twoPeople_withoutAngle(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(poseRaw_bert10_twoPeople_withoutAngle, self).__init__()
        self.hidden_size=100
        self.n_layers=1
        self.attn_heads=1
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
    
        self.bert = BERT5(self.hidden_size,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = x.view(-1,self.length,self.hidden_size)
        input_vectors=x
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
    
class poseRaw2_bert7(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(poseRaw2_bert7, self).__init__()
        self.hidden_size=256
        self.n_layers=1
        self.attn_heads=1
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.2)
        self.one_hot_to_word_embedding = nn.Linear(1024, self.hidden_size)
    
        self.bert = BERT7(self.hidden_size,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
        self.word_embedding_to_one_hot = nn.Linear(self.hidden_size, 1024)
            
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        embedding = self.one_hot_to_word_embedding(x)
        output_embedding , maskSample = self.bert(embedding)
        classificationOut = output_embedding[:,0,:]
        sequenceOut = output_embedding[:,1:,:]
        one_hot_output = self.word_embedding_to_one_hot(sequenceOut)
        output=self.dp(classificationOut)
        fc_out = self.fc_action(output)
        
        return fc_out, one_hot_output
    

    

class poseRaw_bert10X(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(poseRaw_bert10X, self).__init__()
        self.hidden_size=100
        self.n_layers=8
        self.attn_heads=2
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.2)
        
    
        self.bert = BERT5(self.hidden_size,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = x.view(-1,self.length,self.hidden_size)
        input_vectors=x
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample