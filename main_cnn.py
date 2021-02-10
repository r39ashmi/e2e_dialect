#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:01:10 2020

@author: Rashmi Kethireddy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
#device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1024)
class CNN_dialect_model(nn.Module):
    def __init__(self,batch_size,in_channels):
        # Main parameters
        super(CNN_dialect_model, self).__init__()
        self.batch_size=batch_size
        self.in_channels=in_channels
        self.mp1=torch.nn.MaxPool1d(10, stride=10)
        self.cnn1=torch.nn.Conv1d(in_channels=self.in_channels,out_channels=500,kernel_size=5,stride=1,padding=0)      
        self.cnn2=torch.nn.Conv1d(in_channels=500,out_channels=500,kernel_size=3,stride=1,padding=0)
        self.cnn3=torch.nn.Conv1d(in_channels=500,out_channels=3000,kernel_size=5,stride=1,padding=0)
        self.cnn4=torch.nn.Conv1d(in_channels=3000,out_channels=3000,kernel_size=3,stride=1,padding=0)
        self.relu=nn.ReLU()
        self.linear1=nn.Linear(3000,1500)
        self.linear2=nn.Linear(1500,600)
        self.linear3=nn.Linear(600,3)
    def forward(self,data):
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
