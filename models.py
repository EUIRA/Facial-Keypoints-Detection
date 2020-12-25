## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        # Convoultuional layers
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64,128,3)
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        # Batch Normalization    
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm1d(512) # 2D --> 1D
        # Fully Connected Layers
        self.fc1 = nn.Linear(26*26*128,512)
        self.fc2 = nn.Linear(512,136)
        # Dropout       
        self.dropout=nn.Dropout(0.3)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))# Leaky_Relu
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)))) 
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x