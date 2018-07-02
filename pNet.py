#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:25:43 2018

@author: fred
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Tnet (nn.Module):
    def __init__(self, num_point=1024):
        #can be customized, npoints?
        super(Tnet, self).__init__()
        #number of points in each object
        self.num_point = num_point
        #self.batch_size = batch_size
        #height = 1 width = 3
        self.conv1 = nn.Conv2d(1, 64, (1, 3))
        self.conv2 = nn.Conv2d(64, 128, (1, 1))
        self.conv3 = nn.Conv2d(128, 1024, (1, 1))
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc = nn.Linear(256, 9)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcbn2 = nn.BatchNorm1d(256)
    def forward(self, x, is_training=True):
        batch_size = x.size()[0]
        #x = np.expand_dims(x, 1)
        #x.transpose(2,1)
        x = x.view(batch_size, 1, -1, 3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, (self.num_point, 1), (2, 2))
        x = x.view(batch_size, 1024, -1).transpose(2,1)
        x = x.view(batch_size, -1)
        x = F.relu(self.fcbn1(self.fc1(x)))
        x = F.relu(self.fcbn2(self.fc2(x)))
        x = self.fc(x)
        #need a transform and view  
        #iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batch_size,1)
        iden = torch.Tensor([1,0,0,0,1,0,0,0,1])
        iden = Variable(iden.float())
        iden = iden.view(1,9).repeat(batch_size,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x
class Fnet(nn.Module):
    def __init__(self, num_point=1024, global_feature=True):
        super(Fnet, self).__init__()
        self.num_point = num_point
        #self.batch_size = batch_size
        
        self.conv1 = nn.Conv2d(64, 64, (1, 1))
        self.conv2 = nn.Conv2d(64, 128, (1, 1))
        self.conv3 = nn.Conv2d(128, 1024, (1, 1))
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc = nn.Linear(256, 64*64)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcbn2 = nn.BatchNorm1d(256)
    def forward(self, x, is_training=True):
        batch_size = x.size()[0]
        #k = 64
        '''x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, (self.num_point, 1), (2, 2))
        
        x = x.view(batch_size, -1)'''
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, (self.num_point, 1), (2, 2))
        x = x.view(batch_size, 1024, -1).transpose(2,1)
        x = x.view(batch_size, -1)
        x = F.relu(self.fcbn1(self.fc1(x)))
        x = F.relu(self.fcbn2(self.fc2(x)))
        x = self.fc(x)
        #need a transform and view
        #iden = Variable(torch.from_numpy(np.eye(3).flatten().astype(np.float32))).view(1,9).repeat(batch_size,1)
        iden = torch.Tensor(np.eye(64).flatten())
        iden = Variable(iden.float())
        iden = iden.view(1, 64*64).repeat(batch_size,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(batch_size, 64, 64)
        return x
        '''if self.global_feat:
            return x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1)'''
        
class pn_cls(nn.Module):
    def __init__(self, num_point=1024, modelnet10=True):
        super(pn_cls, self).__init__()
        self.num_cat = 10 if modelnet10 else 40
        print(self.num_cat)
        self.num_point = num_point
        self.tnet = Tnet(num_point)
        self.conv1 = nn.Conv2d(1, 64, (1, 3))
        self.conv2 = nn.Conv2d(64, 64, (1, 1))
        self.fnet = Fnet(num_point)
        self.conv3 = nn.Conv2d(64, 64, (1, 1))
        self.conv4 = nn.Conv2d(64, 128, (1, 1))
        self.conv5 = nn.Conv2d(128, 1024, (1, 1))
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_cat)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)
        
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcbn2 = nn.BatchNorm1d(256)
        self.fcbn3 = nn.BatchNorm1d(self.num_cat)
        
    def forward(self, x, is_training = True):
        #print('input_size:', x.size())
        batch_size = x.size()[0]
        num_point = x.size()[1]
        end_points = {}
        trans = self.tnet(x)
        #x = x.transpose(2, 1)
        #x = torch.bmm(x, trans)
        x = torch.matmul(x, trans)
        #x = np.expand_dims(x, -1)
        x = x.view(batch_size, 1, -1, 3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        trans_feature = self.fnet(x)
        end_points['transform'] = trans_feature 
        x = x.view(batch_size, -1, num_point)
        x = torch.matmul(x.transpose(2,1), trans_feature)
        #x = torch.matmul(x.squeeze().transpose(2,1), trans_feature.squeeze())#x.view(batch_size,1, 64, self.num_point).transpose(3, 2), trans_feature)#.view(batch_size,1, 64, 64))
        x = x.transpose(2,1).view(batch_size, 64, self.num_point, 1)
        
        #x.contiguous()
        #x = x.view(batch_size, 64, self.num_point, 1)
        #x=x.view(batch_size, 64, self.num_point, 1)
        #x = np.expand_dims(x, 2)
        #print('PaF_view:',x.size()) 
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, (self.num_point, 1), (2, 2))
        x = x.view(batch_size, -1)
        
        x = F.relu(self.fcbn1(self.fc1(x)))
        x = F.dropout(x, p = 0.7, training = is_training)
        x = F.relu(self.fcbn2(self.fc2(x)))
        x = F.dropout(x, p = 0.7, training = is_training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1), end_points

class pn_cls_basic(nn.Module):
    def __init__(self, num_point=1024, modelnet10=True):
        super(pn_cls_basic, self).__init__()
        self.num_cat = 10 if modelnet10 else 40
        self.num_point = num_point
        self.tnet = Tnet(num_point)
        self.conv1 = nn.Conv2d(1, 64, (1, 3))
        self.conv2 = nn.Conv2d(64, 64, (1, 1))
        #self.fnet = Fnet(num_point)
        self.conv3 = nn.Conv2d(64, 64, (1, 1))
        self.conv4 = nn.Conv2d(64, 128, (1, 1))
        self.conv5 = nn.Conv2d(128, 1024, (1, 1))
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_cat)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)
        
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcbn2 = nn.BatchNorm1d(256)
        self.fcbn3 = nn.BatchNorm1d(self.num_cat)
    def forward(self, x, is_training=True):
        batch_size = x.size()[0]
        end_points = {}
        trans = self.tnet(x)
        x = torch.matmul(x, trans)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, (self.num_point, 1), (2, 2))
        x = x.view(batch_size, -1)
        
        x = F.relu(self.fcbn1(self.fc1(x)))
        x = F.relu(self.fcbn2(self.fc2(x)))
        x = F.dropout(x, p = 0.7, training = is_training)
        #x = F.relu(self.fcbn3(self.fc3(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1), end_points
'''
def weights_init(m):
    if isinstance(m, Tnet):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.xavier_uniform(m.weight.data)
'''     
        
        
        
        
        
        
        
