# -*- coding: utf-8 -*-
"""
Created on Fri May 20 20:04:01 2022

@author: æ´›é”¦
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os


class PDCatNet(nn.Module):
    def __init__(self):
        super(PDCatNet, self).__init__()
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 4, 4),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 4, 4),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 4, 4),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 4, 4),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 4, 4),
            nn.ReLU()
            )
        self.res_conv6 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv7 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv8 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv9 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_convA = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.dtconv1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.dtconv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.dtconv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.dtconv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.dtconv5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.ReLU()
            )
        
        
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))
        
        h = h.cuda()
        c = c.cuda()

        x_list = []
        for i in range(6):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x1 = F.relu(self.res_conv1(x) + resx)
            x2 = F.relu(self.res_conv6(x) + resx)
            x = torch.cat((x1, x2), 1)
            x = F.relu(self.dtconv1(x) + resx)
            
            resx = x
            x1 = F.relu(self.res_conv2(x) + resx)
            x2 = F.relu(self.res_conv7(x) + resx)
            x = torch.cat((x1, x2), 1)
            x = F.relu(self.dtconv2(x) + resx)
            
            resx = x
            x1 = F.relu(self.res_conv3(x) + resx)
            x2 = F.relu(self.res_conv8(x) + resx)
            x = torch.cat((x1, x2), 1)
            x = F.relu(self.dtconv3(x) + resx)
            
            resx = x
            x1 = F.relu(self.res_conv4(x) + resx)
            x2 = F.relu(self.res_conv9(x) + resx)
            x = torch.cat((x1, x2), 1)
            x = F.relu(self.dtconv4(x) + resx)
            
            resx = x
            x1 = F.relu(self.res_conv5(x) + resx)
            x2 = F.relu(self.res_convA(x) + resx)
            x = torch.cat((x1, x2), 1)
            x = F.relu(self.dtconv5(x) + resx)
            
            
            x = self.conv(x)

            x = x + input
            x_list.append(x)

        return x, x_list