import torch
import torch.nn as nn
import numpy as np

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.1)
 
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(5, 5), stride=1, padding=1,)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(16, 16, kernel_size=(5, 5), stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3))
        
        self.conv4 = nn.Conv2d(16, 8, kernel_size=(5, 5), stride=1, padding=1)
        self.act4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()
 
        self.fc3 = nn.Linear(288, 512)
        self.act4 = nn.ReLU()
        self.drop3 = nn.Dropout(0.2)
 
        self.fc4 = nn.Linear(512, 1)
 
    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        
        x = self.act2(self.conv2(x))
        x = self.pool2(x)
        
        x = self.act3(self.conv3(x))
        x = self.pool3(x)
        
        x = self.act4(self.conv4(x))
        x = self.pool4(x)

        x = self.flat(x)
        
        x = self.fc3(x)
        x = self.act4(x)
        x = self.drop3(x)
        
        x = self.fc4(x)
        
        return torch.sigmoid(x)