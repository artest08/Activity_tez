import torch.nn as nn
import torch
import math
from collections import OrderedDict
from .BERT.bert import BERT, BERT2, BERT3, BERT4, BERT5
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'flow_resnet18', 'flow_resnet34', 'flow_resnet50', 'flow_resnet101',
           'flow_resnet152','flow_resnet18_bert3','flow_resnet18_bert4','flow_resnet18_bertX','flow_resnet18_bertX2',
           'flow_resnet18_bert10','flow_resnet18_bert10X','flow_resnet152_bert10', 
           'flow_resnet101_pooling5', 'flow_resnet18_pooling5', 'flow_resnet18_pooling1']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


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

    def __init__(self, block, layers, num_classes=1000, input_frame=20):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_frame, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.dp = nn.Dropout(p=0.7)
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

def change_key_names(old_params, in_channels):
    new_params = OrderedDict()
    layer_count = 0
    allKeyList = old_params.keys()
    for layer_key in allKeyList:
        if layer_count >= len(allKeyList)-2:
            # exclude fc layers
            continue
        else:
            if layer_count == 0:
                rgb_weight = old_params[layer_key]
                # print(type(rgb_weight))
                rgb_weight_mean = torch.mean(rgb_weight, dim=1)
                # TODO: ugly fix here, why torch.mean() turn tensor to Variable
                # print(type(rgb_weight_mean))
                flow_weight = rgb_weight_mean.unsqueeze(1).repeat(1,in_channels,1,1)
                new_params[layer_key] = flow_weight
                layer_count += 1
                # print(layer_key, new_params[layer_key].size(), type(new_params[layer_key]))
            else:
                new_params[layer_key] = old_params[layer_key]
                layer_count += 1
                # print(layer_key, new_params[layer_key].size(), type(new_params[layer_key]))
    
    return new_params

def _load_state_dict(model, model_url,input_frame=2):
    pretrained_dict = model_zoo.load_url(model_url)

    model_dict = model.state_dict()
    new_pretrained_dict = change_key_names(pretrained_dict, input_frame)
    
    new_pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if k in model_dict}
    
    model_dict.update(new_pretrained_dict) 
    
    model.load_state_dict(model_dict)
    

    
    
class flow_resnet18_bert3(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(flow_resnet18_bert3, self).__init__()
        self.frozenFeatureWeights=True
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=16
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features=nn.Sequential(*list(flow_resnet18(pretrained=True).children())[:-3])
        else:
            self.features=nn.Sequential(*list(_trained_flow_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT(512,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(512, num_classes)
            
        if self.frozenFeatureWeights:
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
    
    
class flow_resnet18_pooling1(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(flow_resnet18_pooling1, self).__init__()
        self.featureReduction=512
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.7)
        self.avgpool = nn.AvgPool2d(7)
        self.relu = nn.ReLU(inplace=True)
        self.prelu = nn.PReLU()

        self.features1=nn.Sequential(*list(flow_resnet18(pretrained=True, input_frame=2).children())[:-5])
        self.features2=nn.Sequential(*list(flow_resnet18(pretrained=True, input_frame=2).children())[-5:-3])

        for param in self.features1.parameters():
            param.requires_grad = True

        for param in self.features2.parameters():
            param.requires_grad = True
                
        #self.fc_action = nn.Linear(512, self.featureReduction)
        self.fc_action = nn.Linear(self.featureReduction*self.length, self.num_classes)
        
        
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
    
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.avgpool(x)
        input_and_output = x.view(-1, self.length, 512) 
        x=x.view(-1,self.featureReduction*self.length)
        x = self.dp(x)
        x = self.fc_action(x)
        
        return x, input_and_output, input_and_output, input_and_output
    
class flow_resnet18_bert4(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(flow_resnet18_bert4, self).__init__()
        self.frozenFeatureWeights=False
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        if modelPath=='':
            self.features=nn.Sequential(*list(flow_resnet18(pretrained=True,input_frame=2).children())[:-3])
        else:
            self.features=nn.Sequential(*list(_trained_flow_resnet18(modelPath,num_classes=num_classes).children())[:-3])
        
        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT3(512,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.mapper=nn.Linear(512,512)
        self.fc_action = nn.Linear(512, num_classes)
            
        if self.frozenFeatureWeights:
            for param in self.features.parameters():
                param.requires_grad = False
                
        if self.frozenFeatureWeights:
            for param in self.features.parameters():
                param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
        torch.nn.init.xavier_uniform_(self.mapper.weight)
        self.mapper.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = x.view(-1,self.length,512)
        x=self.mapper(x)
        input_vectors=x
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
    
class flow_resnet18_bert10(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(flow_resnet18_bert10, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.7)

        self.features1=nn.Sequential(*list(flow_resnet18(pretrained=True,input_frame=2).children())[:-5])
        self.features2=nn.Sequential(*list(flow_resnet18(pretrained=True,input_frame=2).children())[-5:-3])
       
        
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
    
    
class flow_resnet152_bert10(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(flow_resnet152_bert10, self).__init__()
        self.hidden_size=2048
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.7)
        
        if modelPath=='':
            self.features1=nn.Sequential(*list(flow_resnet152(pretrained=True,input_frame=2).children())[:-5])
            self.features2=nn.Sequential(*list(flow_resnet152(pretrained=True,input_frame=2).children())[-5:-3])

        self.avgpool = nn.AvgPool2d(7)
        self.bert = BERT5(self.hidden_size,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
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
        x = x.view(-1,self.length,self.hidden_size)
        input_vectors=x
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
    
class flow_resnet18_bert10X(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(flow_resnet18_bert10X, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.7)
        
        if modelPath=='':
            self.features1=nn.Sequential(*list(flow_resnet18(pretrained=True,input_frame=20).children())[:-5])
            self.features2=nn.Sequential(*list(flow_resnet18(pretrained=True,input_frame=20).children())[-5:-3])
        else:
            self.features1=nn.Sequential(*list(_trained_flow_resnet18(modelPath,num_classes=num_classes).children())[:-5])
            self.features2=nn.Sequential(*list(_trained_flow_resnet18(modelPath,num_classes=num_classes).children())[-5:-3])        
        
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
    
    
class flow_resnet101_pooling5(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(flow_resnet101_pooling5, self).__init__()
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        self.avgpool1 = nn.AvgPool2d(28)
        self.avgpool2 = nn.AvgPool2d(14)
        self.avgpool3 = nn.AvgPool2d(7)

        if modelPath=='':
            self.features=nn.Sequential(*list(flow_resnet101(pretrained=True, input_frame = 2).children())[:-6])
            self.features1=nn.Sequential(*list(flow_resnet101(pretrained=True, input_frame = 2).children())[-6])
            self.features2=nn.Sequential(*list(flow_resnet101(pretrained=True, input_frame = 2).children())[-5])
            self.features3=nn.Sequential(*list(flow_resnet101(pretrained=True, input_frame = 2).children())[-4])

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
    
class flow_resnet18_pooling5(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(flow_resnet18_pooling5, self).__init__()
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        self.avgpool1 = nn.AvgPool2d(28)
        self.avgpool2 = nn.AvgPool2d(14)
        self.avgpool3 = nn.AvgPool2d(7)

        if modelPath=='':
            self.features=nn.Sequential(*list(flow_resnet18(pretrained=True, input_frame = 2).children())[:-6])
            self.features1=nn.Sequential(*list(flow_resnet18(pretrained=True, input_frame = 2).children())[-6])
            self.features2=nn.Sequential(*list(flow_resnet18(pretrained=True, input_frame = 2).children())[-5])
            self.features3=nn.Sequential(*list(flow_resnet18(pretrained=True, input_frame = 2).children())[-4])

        for param in self.features.parameters():
            param.requires_grad = True        
        for param in self.features1.parameters():
            param.requires_grad = True
        for param in self.features2.parameters():
            param.requires_grad = True
        for param in self.features3.parameters():
            param.requires_grad = True

        self.fc_action = nn.Linear((128 + 256 + 512)*self.length, self.num_classes)
        
        
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
        features1 = features1.view(-1,self.length,128)
        features2 = features2.view(-1,self.length,256)
        features3 = features3.view(-1,self.length,512)
        #features1 = self.mapper1(features1)
        #features2 = self.mapper2(features2)
        x = torch.cat((features1,features2,features3),2)
        input_and_output = x
        x=x.view(-1,(128 + 256 + 512)*self.length)
        x = self.dp(x)
        x = self.fc_action(x)
        

        return x, input_and_output, input_and_output, input_and_output
    
    
class flow_resnet18_bertX(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(flow_resnet18_bertX, self).__init__()
        self.hidden_size=784
        self.n_layers=1
        self.attn_heads=16
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        self.avgpool = nn.AvgPool2d(8)
        self.bert = BERT5(self.hidden_size,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.avgpool(x)
        x = torch.norm(x, p=2, dim=1)
        x = x.view(x.size(0), -1)
        x = x.view(-1,self.length,self.hidden_size)
        input_vectors=x
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
    
class flow_resnet18_bertX2(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(flow_resnet18_bertX2, self).__init__()
        self.hidden_size=784
        self.n_layers=8
        self.attn_heads=16
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        self.avgpool = nn.AvgPool2d(8)
        self.bert = BERT5(self.hidden_size,length, hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.avgpool(x)
        x = torch.norm(x, p=2, dim=1)
        x = x.view(x.size(0), -1)
        x = x.view(-1,self.length,self.hidden_size)
        input_vectors=x
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample

def flow_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        _load_state_dict(model,model_urls['resnet18'], **kwargs)
    return model


def flow_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
         _load_state_dict(model,model_urls['resnet34'])
    return model


def flow_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_state_dict(model,model_urls['resnet50'])

    return model


def flow_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        _load_state_dict(model,model_urls['resnet101'])
    return model


def flow_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        _load_state_dict(model,model_urls['resnet152'],**kwargs)
    return model

def _trained_flow_resnet18(model_path, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    params = torch.load(model_path)
    pretrained_dict=params['state_dict']
    model.load_state_dict(pretrained_dict)
    return model