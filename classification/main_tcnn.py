#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:01:10 2020

@author: Rashmi Kethireddy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from TCN.tcnn import TemporalConvNet
#device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1024)
class TCNN_dialect(nn.Module):
    def __init__(self,batch_size,in_channels,learnable):
        # Main parameters
        super(TCNN_dialect, self).__init__()
        self.batch_size=batch_size
        self.in_channels=in_channels
        self.mp1=torch.nn.MaxPool1d(10, stride=10)
        self.num_chans1 = [500] * (2 - 1) + [80]
        self.tcnn1=TemporalConvNet(self.in_channels,self.num_chans1,kernel_size=5)
        self.num_chans2 = [500] * (2 - 1) + [500]
        self.tcnn2=TemporalConvNet(80,self.num_chans2,kernel_size=5)
        self.relu=nn.ReLU()
        self.linear1=nn.Linear(500,1500)
        self.linear2=nn.Linear(1500,600)
        self.linear3=nn.Linear(600,3)
        self.drop = nn.Dropout(p=0.51)
        nfilters=40
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,data):
        output1=self.tcnn1(data)
        output1=self.mp1(output1)
        output1=self.tcnn2(output1)
        output1=torch.mean(output1,2)
        output1=self.relu(self.linear1(output1))
        output1=self.relu(self.linear2(output1))
        output1=self.linear3(output1)
        return F.log_softmax(output1,dim=-1)
