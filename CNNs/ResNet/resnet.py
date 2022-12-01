import torch
from torch import nn
from residualblock import *

class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, input_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((2,2))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        layers = []
        
        layers.append(ResBlock(self.in_channels, planes, stride=stride))
        self.in_channels = planes
        
        for i in range(blocks-1):
            layers.append(ResBlock(planes, planes))
        
        print("{}: {}".format(planes, len(layers)))

        return nn.Sequential(*layers)

        
        
def ResNet50(num_classes, channels=3):
    return ResNet(New_ResidualBlock, [3,4,6,3], num_classes, channels)
    
def ResNet101(num_classes, channels=3):
    return ResNet(New_ResidualBlock, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(New_ResidualBlock, [3,8,36,3], num_classes, channels)