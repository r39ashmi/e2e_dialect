#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:01:10 2020

@author: Rashmi Kethireddy
"""
import mel_scale_init as ms_init
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
#device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1024)
class CNN_dialect_model(nn.Module):
    def __init__(self,batch_size,in_channels,learnable):
        # Main parameters
        super(CNN_dialect_model, self).__init__()
        self.batch_size=batch_size
        self.in_channels=in_channels
        self.mp1=torch.nn.MaxPool1d(10, stride=10)
        self.cnn1=torch.nn.Conv1d(in_channels=self.in_channels,out_channels=500,kernel_size=5,stride=1,padding=0)      
        self.cnn2=torch.nn.Conv1d(in_channels=500,out_channels=500,kernel_size=3,stride=1,padding=0)
        self.cnn3=torch.nn.Conv1d(in_channels=500,out_channels=3000,kernel_size=5,stride=1,padding=0)
        self.cnn4=torch.nn.Conv1d(in_channels=3000,out_channels=3000,kernel_size=3,stride=1,padding=0)
        nfilters,fs,nfft=80,8000,1024
        self.mel=learnable
        self.mel_conv=nn.Conv1d(int(nfft/2)+1,nfilters,1,1,padding=0, groups=1, bias=False) # Convolution layer as spectral scale
        freq_points, mel_fbank = ms_init.mel_filter_bank(fs, nfft,nfilters)
        self.mel_conv.weight.data=torch.tensor(mel_fbank,dtype=torch.float32).unsqueeze(-1) # initialized to mel scale
        self.relu=nn.ReLU()
        self.linear1=nn.Linear(3000,1500)
        self.linear2=nn.Linear(1500,600)
        self.linear3=nn.Linear(600,3)
        
    def forward(self,data):
        if int(self.mel) == 1:
            #data=data[:,0:513,:]
            data=self.mel_conv(data)
            data=data.clamp(np.finfo(np.float32).eps)
            data=20*(data.log10())
        output1=self.relu(self.cnn1(data))
        output1=self.relu(self.cnn2(output1))
        output1=self.mp1(output1)
        output1=self.relu(self.cnn3(output1))
        output1=self.relu(self.cnn4(output1))
        output1=torch.mean(output1,2)
        output1=self.relu(self.linear1(output1))
        output1=self.relu(self.linear2(output1))
        output1=self.linear3(output1)
        return F.log_softmax(output1,dim=-1)
