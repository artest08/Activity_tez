import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import sys
from time import time
from .poseNet.poseNet import openPoseL2Part
from .convGRU.convGRU import ConvGRU
from .BERT.bert import BERT3, BERT4, BERT5, BERT6, BERT7
from .TSM.temporal_shift import make_temporal_shift
from .TSM.non_local import make_non_local
from .NLB.NLBlockND import NLBlockND
from .BERT.embedding import BERTEmbedding
import torch
import numpy as np
import itertools as it
import math

__all__ = ['ResNet', 'rgb_resnet18', 'rgb_resnet34', 'rgb_resnet50', 'rgb_resnet101',
           'rgb_resnet152','rgb_openpose_resnet152_type1','rgb_openpose_resnet152_type2'
           ,'rgb_openpose_resnet101_type1','rgb_openpose_resnet101_type3'
           ,'rgb_resnet152_lstm','rgb_resnet152_lstmType2','rgb_resnet152_lstmType3','rgb_resnet152_lstmType4'
           ,'rgb_resnet152_lstmType5',
           'rgb_resnet152_convGRUType1','rgb_resnet18_convGRUType1','rgb_resnet18_convGRUType2'
           ,'rgb_resnet152_pooling1','rgb_resnet18_lstmType6', 'rgb_resnet18_lstmType5','rgb_resnet18_lstmType2','rgb_resnet18_lstmType1',
           'rgb_resnet18_lstmType7', 'rgb_resnet18_bert8', 'rgb_resnet18_bert9',
           'rgb_resnet18_lstmType4','rgb_resnet18_bert10','rgb_resnet18_bert11'
           , 'rgb_resnet18_bert12','rgb_resnet18_bert13','rgb_resnet18_bert14','rgb_resnet18_bert15', 'rgb_resnet18_bert16' 
           ,'rgb_resnet18_pooling1','rgb_resnet18_pooling2','rgb_resnet18_pooling3','rgb_resnet18_pooling4','rgb_resnet18_pooling5','rgb_resnet18_pooling6'
           ,'rgb_resnet18_lstmType3','rgb_resnet152_bert10','rgb_resnet152_lstmType6',
           'rgb_resnet18_bert17','rgb_resnet18_bert10Y',
           'rgb_resnet34_bert10','rgb_resnet50_bert10X','rgb_resnet152_bert10X','rgb_resnet152_bert10XX',
           'rgb_resnet18_NLB10','rgb_resnet18_RankingBert10','rgb_resnet18_RankingBert8','rgb_resnet18_RankingBert8Seg3'
           ,'rgb_resnet18_unpre_bert10', 'rgb_resnet152_bert10_light',
           'rgb_tsm_resnet50', 'rgb_tsm_resnet50_64f', 'rgb_tsm_resnet50_64f_bert10','rgb_tsm_resnet50_8f']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def get_parameter_sizes(model):
    '''Get sizes of all parameters in `model`'''
    mods = list(model.modules())
    sizes = []
    weightNumber=0
    for i in range(1,len(mods)):
        m = mods[i]
        p = list(m.parameters())
        for j in range(len(p)):
            sizes.append(np.array(p[j].size()))
    for s in sizes:
        weightNumber+=np.prod(s)
    return weightNumber 
 
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.dp = nn.Dropout(p=0.8)
        self.fc_action = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dp(x)
        x = self.fc_action(x)

        return x
    
class rgb_openpose_resnet152_type1(nn.Module):
    def __init__(self, num_classes , pretrained=True):
        super(rgb_openpose_resnet152_type1, self).__init__()
        self.openPose=openPoseL2Part()
        self.featureLayer1=nn.Sequential(*list(rgb_resnet152(pretrained=pretrained).children())[:-3])
        self.avgpool = nn.AvgPool2d(7)
        self.avgpool2 = nn.AvgPool2d(4)
        self.dp = nn.Dropout(p=0.8)
        self.fc_action = nn.Linear(512 * 4, num_classes)

        for param in self.openPose.parameters():
            param.requires_grad = False
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.daboth_resnet18_bert10Xa.zero_()
        
    def forward(self, x):
        poses=self.openPose(x)
        poses=torch.abs(poses)
        features=self.featureLayer1(x)
        mask=torch.sum(poses,1,True)
        mask=nn.functional.normalize(mask,dim=1,p=2)
        mask=self.avgpool2(mask)
        #mask=self.maskCreatingLayer(poses)
        x=torch.mul(mask,features)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dp(x)
        x = self.fc_action(x)
        return x
    
class rgb_openpose_resnet101_type1(nn.Module):
    def __init__(self, num_classes , pretrained=True):
        super(rgb_openpose_resnet101_type1, self).__init__()
        self.openPose=openPoseL2Part()
        self.featureLayer1=nn.Sequential(*list(rgb_resnet101(pretrained=pretrained).children())[:-3])
        self.avgpool = nn.AvgPool2d(7)
        self.avgpool2 = nn.AvgPool2d(4)
        self.dp = nn.Dropout(p=0.8)
        self.fc_action = nn.Linear(512 * 4, num_classes)

        for param in self.openPose.parameters():
            param.requires_grad = False
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        poses=self.openPose(x)
        poses=torch.abs(poses)
        poses=poses.view(-1,26,2,28,28)
        poses=torch.norm(poses, p=2, dim=2)
        masks=torch.max(poses,1).values
        masks=self.avgpool2(masks)
        features=self.featureLayer1(x)
        masks=torch.unsqueeze(masks,1)
        #mask=self.maskCreatingLayer(poses)
        x=torch.mul(masks,features)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dp(x)
        x = self.fc_action(x)
        return x
    
    
class rgb_openpose_resnet101_type3(nn.Module):
    def __init__(self, num_classes , pretrained=True):
        super(rgb_openpose_resnet101_type3, self).__init__()
        self.openPose=openPoseL2Part()
        self.featureLayer1=nn.Sequential(*list(rgb_resnet101(pretrained=pretrained).children())[:-3])
        self.avgpool = nn.AvgPool2d(7)
        self.avgpoolPoses = nn.AvgPool2d(4)
        self.dp = nn.Dropout(p=0.15)
        self.fc_action = nn.Linear(512 * 4, num_classes)
        self.fc_attention_1 = nn.Linear(512 * 4, num_classes)
        self.fc_attention_2 = nn.Linear(num_classes, 26)
        self.soft=nn.Softmax(0)
        self.bn=nn.BatchNorm2d(1)
        self.tanh=nn.Tanh()

                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
        torch.nn.init.xavier_uniform_(self.fc_attention_1.weight)
        self.fc_attention_1.bias.data.zero_()

        torch.nn.init.xavier_uniform_(self.fc_attention_2.weight)
        self.fc_attention_2.bias.data.zero_()
        
    def forward(self, x):
        poses=self.openPose(x)
        poses=torch.abs(poses)
        poses=self.avgpoolPoses(poses)
        
        features=self.featureLayer1(x)
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x=self.fc_attention_1(x)
        x=self.tanh(x)
        x=self.fc_attention_2(x)
        x=self.soft(x)
        x=x.view(-1,1).repeat(1,2).view(-1,52,1,1)
        attentionedMasks=torch.mul(x,poses)
        attentionedMasks=torch.sum(attentionedMasks,1,True)  
        attentionedMasks=self.bn(attentionedMasks)
        x=torch.mul(features,attentionedMasks)
        x=self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = self.dp(x)
        x = self.fc_action(x)
        return x




class rgb_openpose_resnet152_type2(nn.Module):
    def __init__(self, num_classes , pretrained=True):
        super(rgb_openpose_resnet152_type2, self).__init__()
        self.openPose=openPoseL2Part()
        self.featureLayer1=nn.Sequential(*list(rgb_resnet152(pretrained=pretrained).children())[:-5])
        self.featureLayer2=nn.Sequential(*list(rgb_resnet152(pretrained=pretrained).children())[-5:-3])
        self.avgpool = nn.AvgPool2d(7)
        self.dp = nn.Dropout(p=0.8)
        self.fc_action = nn.Linear(512 * 4, num_classes)

        for param in self.openPose.parameters():
            param.requires_grad = False
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        poses=self.openPose(x)
        poses=torch.abs(poses)
        features=self.featureLayer1(x)
        mask=torch.sum(poses,1,True)
        mask=nn.functional.normalize(mask,dim=1,p=2)
        #mask=self.maskCreatingLayer(poses)
        x=torch.mul(mask,features)
        x=self.featureLayer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dp(x)
        x = self.fc_action(x)
        return x
    
    
class rgb_resnet152_lstm(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet152_lstm, self).__init__()
        self.frozenFeatureWeights=True
        self.hidden_size=512
        self.num_layers=1
        self.num_classes=num_classes
        self.length=length
        
        if modelPath=='':
            self.features=nn.Sequential(*list(rgb_resnet152(pretrained=False).children())[:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet152(modelPath,num_classes=num_classes).children())[:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.lstm=nn.LSTM(input_size=512 * 4,hidden_size=self.hidden_size, num_layers=self.num_layers,batch_first=True)
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
        if modelPath=='':
            self.trainEnabled=False
        else:
            self.trainEnabled=True
            
        if self.frozenFeatureWeights:
            for param in self.features.parameters():
                param.requires_grad = False
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x=x.view(-1,self.length,512 * 4)
        output,_=self.lstm(x)
        x = self.fc_action(output)
        if self.trainEnabled:
            x=torch.mean(x,1)
        else:
            x=x.view(-1,self.num_classes)
        return x
    
class rgb_resnet152_lstmType2(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet152_lstmType2, self).__init__()
        self.frozenFeatureWeights=True
        self.hidden_size=512
        self.num_layers=2
        self.num_classes=num_classes
        self.length=length
        
        if modelPath=='':
            self.features=nn.Sequential(*list(rgb_resnet152(pretrained=False).children())[:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet152(modelPath,num_classes=num_classes).children())[:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.lstm=nn.LSTM(input_size=512 * 4,hidden_size=self.hidden_size, num_layers=self.num_layers,batch_first=True)
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
        if modelPath=='':
            self.trainEnabled=False
        else:
            self.trainEnabled=True
            
        if self.frozenFeatureWeights:
            for param in self.features.parameters():
                param.requires_grad = False
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x=x.view(-1,self.length,512 * 4)
        output,_=self.lstm(x)
        x = self.fc_action(output)
        if self.trainEnabled:
            x=torch.mean(x,1)
        else:
            x=x.view(-1,self.num_classes)
        return x

class rgb_resnet152_lstmType3(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet152_lstmType3, self).__init__()
        self.frozenFeatureWeights=True
        self.hidden_size=512
        self.num_layers=2
        self.num_classes=num_classes
        self.length=length
        
        if modelPath=='':
            self.features=nn.Sequential(*list(rgb_resnet152(pretrained=False).children())[:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet152(modelPath,num_classes=num_classes).children())[:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.lstm=nn.LSTM(input_size=512 * 4,hidden_size=self.hidden_size, num_layers=self.num_layers,batch_first=True,bidirectional=True)
        self.fc_action = nn.Linear(2*self.hidden_size, num_classes)
            
        if self.frozenFeatureWeights:
            for param in self.features.parameters():
                param.requires_grad = False
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x=x.view(-1,self.length,512 * 4)
        output,_=self.lstm(x)
        x = self.fc_action(output)
        if self.training:
            x=torch.mean(x,1)
        else:
            x=x.view(-1,self.num_classes)
        return x
    
class rgb_resnet18_lstmType3(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_lstmType3, self).__init__()
        self.frozenFeatureWeights=True
        self.hidden_size=512
        self.num_layers=2
        self.num_classes=num_classes
        self.length=length
        
        if modelPath=='':
            self.features=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.lstm=nn.LSTM(input_size=512 ,hidden_size=self.hidden_size, num_layers=self.num_layers,batch_first=True,bidirectional=True)
        self.fc_action = nn.Linear(2*self.hidden_size, num_classes)
            
        if self.frozenFeatureWeights:
            for param in self.features.parameters():
                param.requires_grad = False
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x=x.view(-1,self.length,512)
        output,_=self.lstm(x)
        x = self.fc_action(output)
        if self.training:
            x=torch.mean(x,1)
        else:
            #x=torch.mean(x,1)
            x=x.view(-1,self.num_classes)
        return x
    
class rgb_resnet152_lstmType4(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet152_lstmType4, self).__init__()
        self.frozenFeatureWeights=True
        self.hidden_size=512
        self.num_layers=2
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features=nn.Sequential(*list(rgb_resnet152(pretrained=False).children())[:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet152(modelPath,num_classes=num_classes).children())[:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.lstm=nn.LSTM(input_size=512 * 4,hidden_size=self.hidden_size, num_layers=self.num_layers,batch_first=True,bidirectional=True)
        self.fc_action = nn.Linear(2*self.hidden_size, num_classes)
            
        if self.frozenFeatureWeights:
            for param in self.features.parameters():
                param.requires_grad = False
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x=x.view(-1,self.length,512 * 4)
        output,_=self.lstm(x)
        output= output[:,-1,:]
        output=self.dp(output)
        x = self.dp(x)
        x = self.fc_action(output)
        return x
    
class rgb_resnet152_lstmType6(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet152_lstmType6, self).__init__()
        self.hidden_size=512
        self.num_layers=2
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features1=nn.Sequential(*list(rgb_resnet152(pretrained=True).children())[:-5])
            self.features2=nn.Sequential(*list(rgb_resnet152(pretrained=True).children())[-5:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet152(modelPath,num_classes=num_classes).children())[:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.lstm=nn.LSTM(input_size=512 * 4,hidden_size=self.hidden_size, num_layers=self.num_layers,batch_first=True,bidirectional=True)
        self.fc_action = nn.Linear(2*self.hidden_size, num_classes)
            
        for param in self.features1.parameters():
            param.requires_grad = False
            
        for param in self.features2.parameters():
            param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x=x.view(-1,self.length,512 * 4)
        output,_=self.lstm(x)
        output= output[:,-1,:]
        output=self.dp(output)
        x = self.dp(x)
        x = self.fc_action(output)
        return x
    
class rgb_resnet18_lstmType4(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_lstmType4, self).__init__()
        self.frozenFeatureWeights=True
        self.hidden_size=512
        self.num_layers=2
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.lstm=nn.LSTM(input_size=512,hidden_size=self.hidden_size, num_layers=self.num_layers,batch_first=True,bidirectional=True)
        print(sum(p.numel() for p in self.lstm.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(2*self.hidden_size, num_classes)
            
        if self.frozenFeatureWeights:
            for param in self.features.parameters():
                param.requires_grad = False
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x=x.view(-1,self.length,512)
        output,_=self.lstm(x)
        output= output[:,-1,:]
        output=self.dp(output)
        x = self.fc_action(output)
        return x
    
    
class rgb_resnet18_lstmType2(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_lstmType2, self).__init__()
        self.hidden_size=512
        self.num_layers=2
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features1=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-5])
            self.features2=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[-5:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.lstm=nn.LSTM(input_size=512,hidden_size=self.hidden_size, num_layers=self.num_layers,batch_first=True,bidirectional=True)
        print(sum(p.numel() for p in self.lstm.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(2*self.hidden_size, num_classes)
            
        for param in self.features1.parameters():
            param.requires_grad = False
        for param in self.features2.parameters():
            param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x=x.view(-1,self.length,512)
        output,_=self.lstm(x)
        output= output[:,-1,:]
        output=self.dp(output)
        x = self.fc_action(output)
        return x    
    
    
class rgb_resnet18_lstmType1(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_lstmType1, self).__init__()
        self.hidden_size=512
        self.num_layers=1
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features1=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-5])
            self.features2=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[-5:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.lstm=nn.LSTM(input_size=512,hidden_size=self.hidden_size, num_layers=self.num_layers,batch_first=True,bidirectional=True)
        print(sum(p.numel() for p in self.lstm.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(2*self.hidden_size, num_classes)
            
        for param in self.features1.parameters():
            param.requires_grad = False
        for param in self.features2.parameters():
            param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x=x.view(-1,self.length,512)
        output,_=self.lstm(x)
        output= output[:,-1,:]
        output=self.dp(output)
        x = self.fc_action(output)
        return x  
    
class rgb_resnet18_lstmType7(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_lstmType7, self).__init__()
        self.hidden_size=512
        self.num_layers=3
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features1=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-5])
            self.features2=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[-5:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.lstm=nn.LSTM(input_size=512,hidden_size=self.hidden_size, num_layers=self.num_layers,batch_first=True,bidirectional=True)
        print(sum(p.numel() for p in self.lstm.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(2*self.hidden_size, num_classes)
            
        for param in self.features1.parameters():
            param.requires_grad = False
        for param in self.features2.parameters():
            param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x=x.view(-1,self.length,512)
        output,_=self.lstm(x)
        output= output[:,-1,:]
        output=self.dp(output)
        x = self.fc_action(output)
        return x  
    
class rgb_resnet18_lstmType6(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_lstmType6, self).__init__()
        self.frozenFeatureWeights=True
        self.hidden_size=512
        self.num_layers=2
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.lstm=nn.LSTM(input_size=512,hidden_size=self.hidden_size, num_layers=self.num_layers,batch_first=True,bidirectional=True)
        print(sum(p.numel() for p in self.lstm.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(2*self.hidden_size, num_classes)
        self.mapper=nn.Linear(512,512)
            
        if self.frozenFeatureWeights:
            for param in self.features.parameters():
                param.requires_grad = False
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
        torch.nn.init.xavier_uniform_(self.mapper.weight)
        self.mapper.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x=x.view(-1,self.length,512)
        x=self.mapper(x)
        output,_=self.lstm(x)
        output= output[:,-1,:]
        output=self.dp(output)
        x = self.fc_action(output)
        return x

class rgb_resnet18_RankingBert8(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_RankingBert8, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads = 8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        self.pertutation_matrix, self.possibility_count = self.__create_ordering_matrix()
        
        if modelPath=='':
            self.features1=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-5])
            self.features2=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[-5:-3])
        else:
            self.features1=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-5])
            self.features2=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[-5:-3])        
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT3(512,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(512, num_classes)
        self.ranker = nn.Linear(512 * self.length, self.possibility_count)
            
        for param in self.features1.parameters():
            param.requires_grad = False
        for param in self.features2.parameters():
            param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
    def __create_ordering_matrix(self):
        possibility_count = math.factorial(self.length)
        pertutation_matrix=np.empty((self.length,possibility_count))
        for i,perm in enumerate(it.permutations(range(self.length))):
          pertutation_matrix[:,i] = perm
        return pertutation_matrix, possibility_count
               
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = x.view(-1,self.length,512)
#        input_vectors=x
        batch_size = x.shape[0]
        random_selection_vector_numpy = np.random.randint(self.possibility_count, size = batch_size)
        x = x[np.array(range(batch_size)), 
              self.pertutation_matrix[:,random_selection_vector_numpy],:].permute([1,0,2])
        random_selection_vector_tensor = torch.from_numpy(random_selection_vector_numpy).cuda()
        output , maskSample = self.bert(x)
        output=self.dp(output)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        sequence_tobe_ranked = sequenceOut.view(-1,512 * self.length)
        sequenceRanked = self.ranker(sequence_tobe_ranked)
        x = self.fc_action(classificationOut)
        return x, sequenceRanked, sequenceOut, random_selection_vector_tensor
    
class rgb_resnet18_RankingBert8Seg3(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_RankingBert8Seg3, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads = 8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
#        self.pertutation_matrix = np.array([[1,0,2],[0,1,2],[0,2,1]])
        self.pertutation_matrix, self.possibility_count = self.__create_ordering_matrix()
        
        if modelPath=='':
            self.features1=nn.Sequential(*list(rgb_resnet18(pretrained=False).children())[:-5])
            self.features2=nn.Sequential(*list(rgb_resnet18(pretrained=False).children())[-5:-3])
        else:
            self.features1=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-5])
            self.features2=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[-5:-3])        
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT3(512,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(512, num_classes)
        self.ranker = nn.Linear(512 * self.length, self.possibility_count)
            
        for param in self.features1.parameters():
            param.requires_grad = True
        for param in self.features2.parameters():
            param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def __create_ordering_matrix(self):
        possibility_count = math.factorial(self.length)
        pertutation_matrix=np.empty((self.length,possibility_count))
        for i,perm in enumerate(it.permutations(range(self.length))):
          pertutation_matrix[:,i] = perm
        return pertutation_matrix, possibility_count    
           
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = x.view(-1,self.length,512)
#        input_vectors=x
        batch_size = x.shape[0]
        random_selection_vector_numpy = np.random.randint(self.possibility_count, size = batch_size)
        x = x[np.array(range(batch_size)), 
              self.pertutation_matrix[:,random_selection_vector_numpy],:].permute([1,0,2])
        random_selection_vector_tensor = torch.from_numpy(random_selection_vector_numpy).cuda()
        output , maskSample = self.bert(x)
        output=self.dp(output)
        classificationOut = output[:,0,:]
#        sequenceOut=output[:,1:,:]
        sequenceOut = x
        sequenceOut = sequenceOut.contiguous()
        sequence_tobe_ranked = sequenceOut.view(-1,512 * self.length)
        sequenceRanked = self.ranker(sequence_tobe_ranked)
        x_out = self.fc_action(classificationOut)
        return x_out, sequenceRanked, sequenceOut, random_selection_vector_tensor
    
class rgb_resnet18_bert8(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_bert8, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads = 8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features1=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-5])
            self.features2=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[-5:-3])
        else:
            self.features1=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-5])
            self.features2=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[-5:-3])        
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT3(512,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(512, num_classes)
            
        for param in self.features1.parameters():
            param.requires_grad = False
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
    
    
class rgb_resnet18_bert9(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_bert9, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads = 8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features1=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-5])
            self.features2=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[-5:-3])
        else:
            self.features1=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-5])
            self.features2=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[-5:-3])        
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT4(512,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(512, num_classes)
            
        for param in self.features1.parameters():
            param.requires_grad = False
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

    
class rgb_resnet18_RankingBert10(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_RankingBert10, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads = 8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features1=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-5])
            self.features2=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[-5:-3])
        else:
            self.features1=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-5])
            self.features2=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[-5:-3])        
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT5(512,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(512, num_classes)
        self.ranker = nn.Linear(512, self.length)
            
        for param in self.features1.parameters():
            param.requires_grad = False
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
#        input_vectors=x
        output , maskSample = self.bert(x)
        output=self.dp(output)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        sequenceRanked = self.ranker(sequenceOut)
        x = self.fc_action(classificationOut)
        return x, sequenceRanked, sequenceOut
    
class rgb_resnet18_bert10(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_bert10, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads = 8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features1=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-5])
            self.features2=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[-5:-3])
        else:
            self.features1=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-5])
            self.features2=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[-5:-3])        
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT5(512,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads, mask_prob = 0.75)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(512, num_classes)
            
        for param in self.features1.parameters():
            param.requires_grad = False
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
    
class rgb_resnet18_unpre_bert10(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_unpre_bert10, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads = 8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        

        self.features=nn.Sequential(*list(rgb_resnet18(pretrained=False).children())[:-3])
            
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT5(512,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads, mask_prob = 0.75)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(512, num_classes)
            
        for param in self.features.parameters():
            param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
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

    
class rgb_resnet18_NLB10(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_NLB10, self).__init__()
        self.hidden_size=512
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features1=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-5])
            self.features2=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[-5:-3])
        else:
            self.features1=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-5])
            self.features2=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[-5:-3])        
        self.avgpool = nn.AvgPool3d((self.length, 7, 7), stride=1)
        
        self.NLB = NLBlockND(in_channels = self.hidden_size, inter_channels = self.hidden_size)
        print(sum(p.numel() for p in self.NLB.parameters() if p.requires_grad))
        
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
        #x = self.avgpool(x)
        x = x.view(-1, self.length, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4)
        x = self.NLB(x)
        x = self.avgpool(x)
        x = x.view(-1,self.hidden_size)
        x = self.dp(x)
        x = self.fc_action(x)
        return x
    
class rgb_resnet18_bert10Y(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_bert10Y, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads = 8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        

        self.features1=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-4])
        self.features2=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[-4:-3])
            
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT5(512,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(512, num_classes)
            
        for param in self.features1.parameters():
            param.requires_grad = False
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
    
class rgb_resnet34_bert10(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet34_bert10, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        

        self.features1=nn.Sequential(*list(rgb_resnet34(pretrained=True).children())[:-5])
        self.features2=nn.Sequential(*list(rgb_resnet34(pretrained=True).children())[-5:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT5(512,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(512, num_classes)
            
        for param in self.features1.parameters():
            param.requires_grad = False
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
    
class rgb_resnet50_bert10X(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet50_bert10X, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        

        self.features1=nn.Sequential(*list(rgb_resnet50(pretrained=True).children())[:-5])
        self.features2=nn.Sequential(*list(rgb_resnet50(pretrained=True).children())[-5:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT5(512,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.mapper=nn.Linear(2048,self.hidden_size)
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
        for param in self.features1.parameters():
            param.requires_grad = False
        for param in self.features2.parameters():
            param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mapper(x)
        x = x.view(-1,self.length,512)
        input_vectors=x
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
    
class rgb_resnet152_bert10(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet152_bert10, self).__init__()
        self.hidden_size=2048
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features1=nn.Sequential(*list(rgb_resnet152(pretrained=True).children())[:-5])
            self.features2=nn.Sequential(*list(rgb_resnet152(pretrained=True).children())[-5:-3])
        else:
            self.features1=nn.Sequential(*list(_trained_rgb_resnet152(modelPath,num_classes=num_classes).children())[:-5])
            self.features2=nn.Sequential(*list(_trained_rgb_resnet152(modelPath,num_classes=num_classes).children())[-5:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT5(self.hidden_size,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
        for param in self.features1.parameters():
            param.requires_grad = False
        for param in self.features2.parameters():
            param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
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
    
class rgb_resnet152_bert10_light(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet152_bert10_light, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features1=nn.Sequential(*list(rgb_resnet152(pretrained=True).children())[:-5])
            self.features2=nn.Sequential(*list(rgb_resnet152(pretrained=True).children())[-5:-3])
        else:
            self.features1=nn.Sequential(*list(_trained_rgb_resnet152(modelPath,num_classes=num_classes).children())[:-5])
            self.features2=nn.Sequential(*list(_trained_rgb_resnet152(modelPath,num_classes=num_classes).children())[-5:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT5(self.hidden_size,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
        self.reduction = nn.Linear(self.hidden_size * 4, self.hidden_size)
            
        for param in self.features1.parameters():
            param.requires_grad = False
        for param in self.features2.parameters():
            param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.reduction(x)
        x = x.view(-1,self.length,self.hidden_size)
        input_vectors=x
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
    
    
class rgb_resnet152_bert10X(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet152_bert10X, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features1=nn.Sequential(*list(rgb_resnet152(pretrained=True).children())[:-5])
            self.features2=nn.Sequential(*list(rgb_resnet152(pretrained=True).children())[-5:-3])
        else:
            self.features1=nn.Sequential(*list(_trained_rgb_resnet152(modelPath,num_classes=num_classes).children())[:-5])
            self.features2=nn.Sequential(*list(_trained_rgb_resnet152(modelPath,num_classes=num_classes).children())[-5:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT5(self.hidden_size,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.mapper=nn.Linear(2048,self.hidden_size)
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
        for param in self.features1.parameters():
            param.requires_grad = False
        for param in self.features2.parameters():
            param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mapper(x)
        x = x.view(-1,self.length,self.hidden_size)
        input_vectors=x
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
    
class rgb_resnet152_bert10XX(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet152_bert10XX, self).__init__()
        self.hidden_size=2048
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features1=nn.Sequential(*list(rgb_resnet152(pretrained=True).children())[:-4])
            self.features2=nn.Sequential(*list(rgb_resnet152(pretrained=True).children())[-4:-3])
        else:
            self.features1=nn.Sequential(*list(_trained_rgb_resnet152(modelPath,num_classes=num_classes).children())[:-4])
            self.features2=nn.Sequential(*list(_trained_rgb_resnet152(modelPath,num_classes=num_classes).children())[-4:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT5(self.hidden_size,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
        for param in self.features1.parameters():
            param.requires_grad = False
        for param in self.features2.parameters():
            param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
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
    
class rgb_resnet18_bert11(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_bert11, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT5(512,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(512, num_classes)
            
        for param in self.features.parameters():
            param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
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
    
    
class rgb_resnet18_bert12(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_bert12, self).__init__()
        self.hidden_size=512
        self.n_layers=8
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features1=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-5])
            self.features2=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[-5:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT5(512,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(512, num_classes)
            
        for param in self.features1.parameters():
            param.requires_grad = False
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
    
class rgb_resnet18_bert13(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_bert13, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT5(512,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(512, num_classes)
            
        for param in self.features.parameters():
            param.requires_grad = False
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
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
    
    
class rgb_resnet18_bert14(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_bert14, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features1=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-5])
            self.features2=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[-5:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        
        self.bert_feature = BERT5(512,49, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        
        self.bert = BERT5(512,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))

        self.fc_action = nn.Linear(512, num_classes)
            
        for param in self.features1.parameters():
            param.requires_grad = False
        for param in self.features2.parameters():
            param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x=x.view(x.size(0),512,-1)
        x=x.transpose(1,2)
        input_vectors=x
        extracted_spatial_attention_features , maskSample = self.bert_feature(x)
        feature_out = extracted_spatial_attention_features[:,0,:]
        sequenceOut=extracted_spatial_attention_features[:,1:,:]
        x = feature_out.view(-1,self.length,512)       
        output , _ = self.bert(x)
        classificationOut = output[:,0,:]
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
    
    
class rgb_resnet18_bert15(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_bert15, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features1=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-4])
            self.features2=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[-4:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT5(512,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(512, num_classes)
            
        for param in self.features1.parameters():
            param.requires_grad = False
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
    
class rgb_resnet18_bert16(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_bert16, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        assert self.length % 4 == 0, "Length is not convenient"
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features1=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-5])
            self.features2=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[-5:-3])
        else:
            self.features1=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-5])
            self.features2=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[-5:-3])        
        self.avgpool = nn.AvgPool2d(7)
        self.bert1 = BERT5(512,int (length/4), hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        self.bert2 = BERT5(512,int (length/4), hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        self.bert3 = BERT5(512,int (length/4), hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        self.bert4 = BERT5(512,int (length/4), hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert1.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(512, num_classes)
            
        for param in self.features1.parameters():
            param.requires_grad = False
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
        output1 , maskSample = self.bert1(x[:,:4,:])
        output2 , maskSample = self.bert2(x[:,4:8,:])
        output3 , maskSample = self.bert3(x[:,8:12,:])
        output4 , maskSample = self.bert3(x[:,12:16,:])
        classificationOut1 = output1[:,0,:]
        classificationOut2 = output2[:,0,:]
        classificationOut3 = output3[:,0,:]
        classificationOut4 = output4[:,0,:]
        classificationOut = (classificationOut1 +classificationOut2 +  classificationOut3 + classificationOut4) / 4
        sequenceOut1=output1[:,1:,:]
        sequenceOut2=output2[:,1:,:]
        sequenceOut3=output3[:,1:,:]
        sequenceOut4=output4[:,1:,:]
        sequenceOut = torch.cat((sequenceOut1,sequenceOut2,sequenceOut3,sequenceOut4),1)
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample    

class rgb_resnet18_bert17(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_bert17, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        self.avgpool = nn.AvgPool2d(7)
        if modelPath=='':
            self.features1=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-5])
            self.features2=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[-5:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        
        self.bert_feature = BERT6(512,49, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=1)
        print(sum(p.numel() for p in self.bert_feature.parameters() if p.requires_grad))
        self.bert1 = BERT5(512,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=1)
        self.bert2 = BERT6(512,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        
        print(sum(p.numel() for p in self.bert1.parameters() if p.requires_grad))

        self.fc_action = nn.Linear(512, num_classes)
            
        for param in self.features1.parameters():
            param.requires_grad = False
        for param in self.features2.parameters():
            param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        features=x.view(x.size(0),512,-1)
        x = self.avgpool(x)
        x = x.view(-1, self.length, 512)
        features=features.transpose(1,2)
        summary_tobe_extracted , _ = self.bert1(x)
        summary = torch.unsqueeze(summary_tobe_extracted[:,0,:],1)
        extracted_spatial_attention_features , maskSample = self.bert_feature(features,summary.repeat(16,1,1))
        feature_out = extracted_spatial_attention_features[:,0,:]
        x = feature_out.view(-1,self.length,512) 
        input_vectors=x
        output , _ = self.bert2(x, summary)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
class rgb_resnet152_lstmType5(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet152_lstmType5, self).__init__()
        self.frozenFeatureWeights=True
        self.hidden_size=512
        self.num_layers=2
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        
        if modelPath=='':
            self.features=nn.Sequential(*list(rgb_resnet152(pretrained=False).children())[:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet152(modelPath,num_classes=num_classes).children())[:-3])
        
        self.openPose=openPoseL2Part()
        self.avgpool = nn.AvgPool2d(7)
        self.maxpool = nn.MaxPool2d(4)
        self.lstm=nn.LSTM(input_size=512 * 4,hidden_size=self.hidden_size, num_layers=self.num_layers,batch_first=True,bidirectional=True)
        self.fc_action = nn.Linear(2*self.hidden_size, num_classes)
            
        if self.frozenFeatureWeights:
            for param in self.features.parameters():
                param.requires_grad = False
            for param in self.openPose.parameters():
                param.requires_grad = False
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
    
    def forward(self, x):
        poses=self.openPose(x)
        poses=torch.abs(poses)
        poses=poses.view(-1,26,2,28,28)
        poses=torch.norm(poses, p=2, dim=2)
        masks=torch.max(poses,1).values
        masks=self.maxpool(masks)
        
        features=self.features(x)
        masks=(masks.view(-1,49)/torch.norm(masks.view(-1,49), p=10, dim=1).unsqueeze(1)).view(-1,7,7)
        masks=torch.unsqueeze(masks,1)
        #mask=self.maskCreatingLayer(poses)
        x=torch.mul(masks,features)
        
        x = self.avgpool(x)
        x=x.view(-1,self.length,512 * 4)
        output,_=self.lstm(x)
        x= output[:,-1,:]
        x = self.dp(x)
        x = self.fc_action(x)
        
        return x   

class rgb_resnet18_lstmType5(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_lstmType5, self).__init__()
        self.frozenFeatureWeights=True
        self.hidden_size=512
        self.num_layers=2
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        
        if modelPath=='':
            self.features=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        
        self.openPose=openPoseL2Part()
        self.avgpool = nn.AvgPool2d(7)
        self.maxpool = nn.MaxPool2d(4)
        self.lstm=nn.LSTM(input_size=512,hidden_size=self.hidden_size, num_layers=self.num_layers,batch_first=True,bidirectional=True)
        self.fc_action = nn.Linear(2*self.hidden_size, num_classes)
            
        if self.frozenFeatureWeights:
            for param in self.features.parameters():
                param.requires_grad = False
            for param in self.openPose.parameters():
                param.requires_grad = False
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
    
    def forward(self, x):
        poses=self.openPose(x)
        poses=torch.abs(poses)
        poses=poses.view(-1,26,2,28,28)
        poses=torch.norm(poses, p=2, dim=2)
        masks=torch.max(poses,1).values
        masks=self.maxpool(masks)
        
        features=self.features(x)
        masks=(masks.view(-1,49)/torch.norm(masks.view(-1,49), p=10, dim=1).unsqueeze(1)).view(-1,7,7)
        masks=torch.unsqueeze(masks,1)
        #mask=self.maskCreatingLayer(poses)
        x=torch.mul(masks,features)
        
        x = self.avgpool(x)
        x=x.view(-1,self.length,512)
        output,_=self.lstm(x)
        x= output[:,-1,:]
        x = self.dp(x)
        x = self.fc_action(x)
        
        return x  

class rgb_resnet152_pooling1(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet152_pooling1, self).__init__()
        self.frozenFeatureWeights=True
        self.featureReduction=128
        self.num_layers=2
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        self.avgpool = nn.AvgPool2d(7)
        self.relu = nn.ReLU(inplace=True)
        
        if modelPath=='':
            self.features=nn.Sequential(*list(rgb_resnet152(pretrained=False).children())[:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet152(modelPath,num_classes=num_classes).children())[:-3])
        if self.frozenFeatureWeights:
            for param in self.features.parameters():
                param.requires_grad = False
                
        self.fc_action = nn.Linear(2048, self.featureReduction)
        self.fc_action2 = nn.Linear(self.featureReduction*self.length, self.num_classes)
        
        
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
    
    def forward(self, x):
        features=self.features(x)
        features = self.avgpool(features)
        features=features.view(-1,self.length,2048)
        x = self.fc_action(features)
        x = self.relu(x)
        xNorm = torch.norm(x,p=2,dim=2)
        x_diff = xNorm[:,1:] - xNorm[:,:-1] - 0.2
        x_diff = self.relu(x_diff)
        x=x.view(-1,self.featureReduction*self.length)
        x = self.dp(x)
        x = self.fc_action2(x)
        
        
        return x,x_diff  

class rgb_resnet18_pooling1(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_pooling1, self).__init__()
        self.frozenFeatureWeights=True
        self.featureReduction=128
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        self.avgpool = nn.AvgPool2d(7)
        self.relu = nn.ReLU(inplace=True)
        self.prelu = nn.PReLU()
        if modelPath=='':
            self.features=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        if self.frozenFeatureWeights:
            for param in self.features.parameters():
                param.requires_grad = False
        else:
            for param in self.features.parameters():
                param.requires_grad = True
                
        self.fc_action = nn.Linear(512, self.featureReduction)
        self.fc_action2 = nn.Linear(self.featureReduction*self.length, self.num_classes)
        
        
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
    
    def forward(self, x):
        features=self.features(x)
        features = self.avgpool(features)
        features=features.view(-1,self.length,512)
        x = self.fc_action(features)
        x = self.prelu(x)
        xNorm = torch.norm(x,p=2,dim=2)
        x_diff = xNorm[:,1:] - xNorm[:,:-1] - 0.2
        x_diff = self.relu(x_diff)
        x=x.view(-1,self.featureReduction*self.length)
        x = self.dp(x)
        x = self.fc_action2(x)
        
        if self.training:
            return x,x_diff     
        else:
            return x
        
class rgb_resnet18_pooling2(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_pooling2, self).__init__()
        self.frozenFeatureWeights=True
        self.featureReduction=32
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        self.avgpool = nn.AvgPool2d(7)
        self.relu = nn.ReLU(inplace=True)
        self.prelu = nn.PReLU()
        if modelPath=='':
            self.features=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        if self.frozenFeatureWeights:
            for param in self.features.parameters():
                param.requires_grad = False
                
        self.fc_action = nn.Linear(512, self.featureReduction)
        self.fc_action2 = nn.Linear(self.featureReduction*self.length, self.num_classes)
        
        
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
    
    def forward(self, x):
        features=self.features(x)
        features = self.avgpool(features)
        features=features.view(-1,self.length,512)
        x = self.fc_action(features)
        x = self.prelu(x)
        xNorm = torch.norm(x,p=2,dim=2)
        x_diff = xNorm[:,1:] - xNorm[:,:-1] - 0.2
        x_diff = self.relu(x_diff)
        x=x.view(-1,self.featureReduction*self.length)
        x = self.dp(x)
        x = self.fc_action2(x)
        
        if self.training:
            return x,x_diff     
        else:
            return x
        
class rgb_resnet18_pooling3(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_pooling3, self).__init__()
        self.frozenFeatureWeights=True
        self.featureReduction=512
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        self.avgpool = nn.AvgPool2d(7)
        self.relu = nn.ReLU(inplace=True)
        self.prelu = nn.PReLU()
        if modelPath=='':
            self.features=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        if self.frozenFeatureWeights:
            for param in self.features.parameters():
                param.requires_grad = False
                
        self.fc_action = nn.Linear(512, self.featureReduction)
        self.fc_action2 = nn.Linear(self.featureReduction*self.length, self.num_classes)
        
        
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
    
    def forward(self, x):
        features=self.features(x)
        features = self.avgpool(features)
        features=features.view(-1,self.length,512)
        x = self.fc_action(features)
        x = self.prelu(x)
        xNorm = torch.norm(x,p=2,dim=2)
        x_diff = xNorm[:,1:] - xNorm[:,:-1] - 0.2
        x_diff = self.relu(x_diff)
        x=x.view(-1,self.featureReduction*self.length)
        x = self.dp(x)
        x = self.fc_action2(x)
        
        if self.training:
            return x,x_diff     
        else:
            return x
        
class rgb_resnet18_pooling4(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_pooling4, self).__init__()
        self.frozenFeatureWeights=True
        self.featureReduction=512
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        self.avgpool = nn.AvgPool2d(7)
        self.relu = nn.ReLU(inplace=True)
        self.prelu = nn.PReLU()
        if modelPath=='':
            self.features=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        if self.frozenFeatureWeights:
            for param in self.features.parameters():
                param.requires_grad = False
                
        self.fc_action = nn.Linear(512, self.featureReduction)
        self.fc_action2 = nn.Linear(self.featureReduction*(self.length-1), self.num_classes)
        
        
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
    
    def forward(self, x):
        features=self.features(x)
        features = self.avgpool(features)
        features=features.view(-1,self.length,512)
        features=features[:,1:,:]-features[:,:-1,:]
        x = self.fc_action(features)
        x = self.prelu(x)
        x=x.view(-1,self.featureReduction*(self.length-1))
        x = self.dp(x)
        x = self.fc_action2(x)
        
        if self.training:
            return x,None     
        else:
            return x
        
class rgb_resnet18_pooling5(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_pooling5, self).__init__()
        self.frozenFeatureWeights=True
        self.featureReduction=128
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        self.avgpool = nn.AvgPool2d(7)
        self.relu = nn.ReLU(inplace=True)
        self.prelu = nn.PReLU()
        self.embedding = BERTEmbedding(input_dim=self.featureReduction, max_len=self.length)
        if modelPath=='':
            self.features=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        if self.frozenFeatureWeights:
            for param in self.features.parameters():
                param.requires_grad = False
                
        self.fc_action = nn.Linear(512, self.featureReduction)
        self.fc_action2 = nn.Linear(self.featureReduction*self.length, self.num_classes)
        
        
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
    
    def forward(self, x):
        features=self.features(x)
        features = self.avgpool(features)
        features=features.view(-1,self.length,512)
        x = self.fc_action(features)
        x = self.prelu(x)
        x = self.embedding(x)
        x=x.view(-1,self.featureReduction*self.length)
        x = self.dp(x)
        x = self.fc_action2(x)
        
        if self.training:
            return x,None     
        else:
            return x
        
class rgb_resnet18_pooling6(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_pooling6, self).__init__()
        self.frozenFeatureWeights=True
        self.featureReduction1=64
        self.featureReduction2=128
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        self.avgpool1 = nn.AvgPool2d(14)
        self.avgpool2 = nn.AvgPool2d(7)

        if modelPath=='':
            self.features1=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-4])
            self.features2=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[-4])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        
        if self.frozenFeatureWeights:
            for param in self.features1.parameters():
                param.requires_grad = False
            for param in self.features2.parameters():
                param.requires_grad = False
                
        self.mapper1 = nn.Sequential(nn.Linear(256, self.featureReduction1),nn.PReLU())
        self.mapper2 = nn.Sequential(nn.Linear(512, self.featureReduction2),nn.PReLU())
        
        self.fc_action = nn.Linear((self.featureReduction1+self.featureReduction2)*self.length, self.num_classes)
        
        
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
    
    def forward(self, x):
        features1 = self.features1(x)
        features2 = self.features2(features1)
        features1 = self.avgpool1(features1)
        features2 = self.avgpool1(features2)
        features1 = features1.view(-1,self.length,256)
        features2 = features2.view(-1,self.length,512)
        features1 = self.mapper1(features1)
        features2 = self.mapper2(features2)
        x = torch.cat((features1,features2),2)
        x=x.view(-1,(self.featureReduction1+self.featureReduction2)*self.length)
        x = self.dp(x)
        x = self.fc_action(x)
        
        if self.training:
            return x,None     
        else:
            return x

class rgb_resnet152_convGRUType1(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet152_convGRUType1, self).__init__()
        self.frozenFeatureWeights=True
        self.hidden_size=512
        self.num_layers=2
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        
        if modelPath=='':
            self.features=nn.Sequential(*list(rgb_resnet152(pretrained=False).children())[:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet152(modelPath,num_classes=num_classes).children())[:-3])
        
        self.openPose=openPoseL2Part()
        self.avgpool = nn.AvgPool2d(7)
        self.avgpool2 = nn.AvgPool2d(4)
        
        self.ConvGRU=ConvGRU(input_size=512 * 4,hidden_sizes=self.hidden_size,kernel_sizes=3,n_layers=self.length)
        self.fc_action = nn.Linear(self.hidden_size*49, num_classes)
            
        if self.frozenFeatureWeights:
            for param in self.features.parameters():
                param.requires_grad = False

                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x=x.view(-1,self.length,512 * 4,7,7)
        output=self.ConvGRU(x)
        output= output[:,-1]
        output=output.view(-1,self.hidden_size*49)
        x = self.dp(x)
        x = self.fc_action(output)
        return x
    
class rgb_resnet18_convGRUType1(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_convGRUType1, self).__init__()
        self.frozenFeatureWeights=True
        self.hidden_size=64
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        
        if modelPath=='':
            self.features=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        
        #self.openPose=openPoseL2Part()
        self.avgpool = nn.AvgPool2d(7)
        self.avgpool2 = nn.AvgPool2d(4)
        
        self.ConvGRU=ConvGRU(input_size=512,hidden_sizes=self.hidden_size,kernel_sizes=3,n_layers=self.length)
        print(sum(p.numel() for p in self.ConvGRU.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size*49, num_classes)
            
        if self.frozenFeatureWeights:
            for param in self.features.parameters():
                param.requires_grad = False

                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x=x.view(-1,self.length,512,7,7)
        output=self.ConvGRU(x)
        output= output[:,-1]
        output=output.view(-1,self.hidden_size*49)
        x = self.dp(x)
        x = self.fc_action(output)
        return x

class rgb_resnet18_convGRUType2(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_resnet18_convGRUType2, self).__init__()
        self.hidden_size=196
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        
        if modelPath=='':
            self.features1=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[:-5])
            self.features2=nn.Sequential(*list(rgb_resnet18(pretrained=True).children())[-5:-3])
        else:
            self.features=nn.Sequential(*list(_trained_rgb_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        
        #self.openPose=openPoseL2Part()
        self.avgpool = nn.AvgPool2d(7)
        self.avgpool2 = nn.AvgPool2d(4)
        
        self.ConvGRU=ConvGRU(input_size=512,hidden_sizes=self.hidden_size,kernel_sizes=3,n_layers=self.length)
        print(sum(p.numel() for p in self.ConvGRU.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size*49, num_classes)
            
        for param in self.features1.parameters():
            param.requires_grad = False
        for param in self.features2.parameters():
            param.requires_grad = True

                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x=x.view(-1,self.length,512,7,7)
        output=self.ConvGRU(x)
        output= output[:,-1]
        output=output.view(-1,self.hidden_size*49)
        x = self.dp(x)
        x = self.fc_action(output)
        return x


        
def _load_state_dict(model, model_url):
    # '.'s are no longer allowed in module names, but pervious _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pretrained_dict = model_zoo.load_url(model_url)

    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    
def rgb_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        _load_state_dict(model,model_urls['resnet18'])
    return model


def rgb_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
         _load_state_dict(model,model_urls['resnet34'])
    return model


def rgb_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_state_dict(model,model_urls['resnet50'])

    return model


def rgb_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        _load_state_dict(model,model_urls['resnet101'])
    return model


def rgb_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        _load_state_dict(model,model_urls['resnet152'])
    return model

def _trained_rgb_resnet152(model_path, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    params = torch.load(model_path)
    pretrained_dict=params['state_dict']
    model.load_state_dict(pretrained_dict)
    return model

def _trained_rgb_resnet18(model_path, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    params = torch.load(model_path)
    pretrained_dict=params['state_dict']
    model.load_state_dict(pretrained_dict)
    return model

def rgb_tsm_resnet50(modelPath=''):
    num_segments = 8
    shift_div = 8
    shift_place = 'blockres'
    temporal_pool = False
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=400)    
    make_temporal_shift(model, num_segments,
        n_div=shift_div, place=shift_place, temporal_pool=temporal_pool)
    
    if modelPath != '':
        params = torch.load(modelPath)
        kinetics_dict = params['state_dict']
        
        model_dict = model.state_dict()
        kinetics_dict_new = {}
    
        # 1. filter out unnecessary keys
        for k, v in kinetics_dict.items():
            if 'module.base_model.' in k:
                kinetics_dict_new.update({k[18:]:v})
    
        # 2. overwrite entries in the existing state dict
        model_dict.update(kinetics_dict_new) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model

def rgb_tsm_resnet50NL(modelPath=''):
    num_segments = 8
    shift_div = 8
    shift_place = 'blockres'
    temporal_pool = False
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=400)    
    make_temporal_shift(model, num_segments,
        n_div=shift_div, place=shift_place, temporal_pool=temporal_pool)
    
    make_non_local(model, num_segments)
    if modelPath != '':
        params = torch.load(modelPath)
        kinetics_dict = params['state_dict']
        
        model_dict = model.state_dict()
        kinetics_dict_new = {}
    
        # 1. filter out unnecessary keys
        for k, v in kinetics_dict.items():
            if 'module.base_model.' in k:
                kinetics_dict_new.update({k[18:]:v})
    
        # 2. overwrite entries in the existing state dict
        model_dict.update(kinetics_dict_new) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model



class rgb_tsm_resnet50_64f(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_tsm_resnet50_64f, self).__init__()
        self.dp = nn.Dropout(p=0.8)
        self.length = 64
        self.avgpool = nn.AvgPool2d(7)
        self.tsm_selection = np.array(range(0, 64, 8)) + 4
    
        self.features=nn.Sequential(*list(rgb_tsm_resnet50(modelPath).children())[:-3])
        self.fc_action = nn.Linear(2048, num_classes)
            
        for param in self.features.parameters():
            param.requires_grad = True

                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = x.view(-1, self.length, 3, 224, 224)
        x = x[:, self.tsm_selection]
        x = x.view(-1, 3, 224, 224)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(-1, 2048) 
        x = self.dp(x)
        output = self.fc_action(x)
        output = output.view(-1,8,output.shape[1])
        output = torch.mean(output, 1)
        return output
    
class rgb_tsm_resnet50_8f(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_tsm_resnet50_8f, self).__init__()
        self.dp = nn.Dropout(p=0.8)
        self.length = 8
        self.avgpool = nn.AvgPool2d(7)
    
        self.features=nn.Sequential(*list(rgb_tsm_resnet50(modelPath).children())[:-3])
        self.fc_action = nn.Linear(2048, num_classes)
            
        for param in self.features.parameters():
            param.requires_grad = True

                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(-1, 2048) 
        x = self.dp(x)
        output = self.fc_action(x)
        output = output.view(-1,8,output.shape[1])
        output = torch.mean(output, 1)
        return output
    
    
class rgb_tsm_resnet50_seg(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_tsm_resnet50_seg, self).__init__()
        self.dp = nn.Dropout(p=0.8)
        self.length = length
        self.avgpool = nn.AvgPool2d(7)
        self.tsm_selection = np.array(range(0, 64, 8)) + 4
    
        self.features=nn.Sequential(*list(rgb_tsm_resnet50(modelPath).children())[:-3])
        self.fc_action = nn.Linear(2048, num_classes)
            
        for param in self.features.parameters():
            param.requires_grad = True

                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = x.view(-1, self.length, 3, 224, 224)
        x = x[:, self.tsm_selection]
        x = x.view(-1, 3, 224, 224)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(-1, 2048) 
        x = self.dp(x)
        output = self.fc_action(x)
        output = output.view(-1,8,output.shape[1])
        output = torch.mean(output, 1)
        return output
    
class rgb_tsm_resnet50_64f_bert10(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_tsm_resnet50_64f_bert10, self).__init__()
        self.hidden_size=2048
        self.n_layers=1
        self.attn_heads=8
        self.dp = nn.Dropout(p=0.8)
        self.avgpool = nn.AvgPool2d(7)
        self.tsm_selection = np.array(range(0, 64, 8)) + 4
    
        self.bert = BERT5(self.hidden_size, 8, hidden=self.hidden_size, 
                          n_layers=self.n_layers, attn_heads=self.attn_heads)
        
        self.features=nn.Sequential(*list(rgb_tsm_resnet50(modelPath).children())[:-3])
        self.fc_action = nn.Linear(2048, num_classes)
            
        for param in self.features.parameters():
            param.requires_grad = True

                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = x.view(-1, 64, 3, 224, 224)
        x = x[:, self.tsm_selection]
        x = x.view(-1, 3, 224, 224)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(-1, self.hidden_size) 
        x = x.view(-1, 8, self.hidden_size)
        input_vectors=x
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
    
    


