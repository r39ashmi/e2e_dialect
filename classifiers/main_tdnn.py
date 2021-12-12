#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:01:10 2020

@author: Rashmi Kethireddy
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from TDNN.tdnn import TDNN
#device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1024)
class TDNN_dialect(nn.Module):
    def __init__(self,batch_size,in_channels,learnable):
        # Main parameters
        super(TDNN_dialect, self).__init__()
        self.batch_size=batch_size
        self.in_channels=in_channels
        self.frame1 = TDNN(input_dim=self.in_channels, output_dim=512, context_size=5, dilation=1)
        self.frame2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2)
        self.frame3 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3)
        self.frame4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
        self.frame5 = TDNN(input_dim=512, output_dim=1500, context_size=1, dilation=1)
        self.relu=nn.ReLU()
        self.linear1=nn.Linear(1500,1500)
        self.linear2=nn.Linear(1500,600)
        self.linear3=nn.Linear(600,3)
        nfilters=40
    def forward(self,data):
        output1=self.frame1(data)
        output1=self.frame2(output1)
        output1=self.frame3(output1)
        output1=self.frame4(output1)
        output1=self.frame5(output1)
        output1=output1.permute(0,2,1)
        output1=torch.mean(output1,2)
        output1=self.relu(self.linear1(output1))
        output1=self.relu(self.linear2(output1))
        output1=self.linear3(output1)
        return F.log_softmax(output1,dim=-1)
