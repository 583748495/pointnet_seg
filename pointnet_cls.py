#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 23:00:07 2018

@author: fred
"""

import argparse

import numpy as np
import os
import sys
import visdom
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from pNet import pn_cls
from pNet import pn_cls_basic
from modelnet_normal import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log_trans3D_transpose_10_1000', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=1000, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--workers', type=int, default=4, help='number of workers used to load data [default:4]')
parser.add_argument('--modelnet10', type = bool, default=True, help='Dataset: use modelnet10 or modelnet40')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
NUM_WORKER = FLAGS.workers
MODELNET10 = FLAGS.modelnet10

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
#MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
os.system('cp %s %s' % (BASE_DIR, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def main():

    '''
    BASE_LEARNING_RATE = FLAGS.learning_rate
    GPU_INDEX = FLAGS.gpu
    MOMENTUM = FLAGS.momentum
    OPTIMIZER = FLAGS.optimizer
    DECAY_STEP = FLAGS.decay_step
    DECAY_RATE = FLAGS.decay_rate
    MODEL = importlib.import_module(FLAGS.model) # import network module
    '''
    
    net = pn_cls(modelnet10=MODELNET10)
    #net = pn_cls_basic(modelnet10=MODELNET10)
    dsDir = '/home/lyx/data/modelnet40_normal_resampled/'
    #dsDir = '/home/fred/lyx/data/modelnet40_normal_resampled/'
    trainset = Dataset(dsDir, batch_size = BATCH_SIZE, npoints = NUM_POINT, shuffle = True, modelnet10=MODELNET10)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=NUM_WORKER)
    
    testset = Dataset(dsDir, batch_size=BATCH_SIZE, npoints=NUM_POINT, kind='test', modelnet10=MODELNET10)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, num_workers=NUM_WORKER)

    vis = visdom.Visdom(port=8197) 
    #criterion = nn.CrossEntropyLoss() 
    #criterion = nn.NLLLOSS()
    #MSEcri = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 40, 0.7)
    net.cuda()
    
    x, l1, l2 = 0, 0, 0
    win = vis.line(X=np.column_stack((np.array([x]),np.array([x]))), Y=np.column_stack((np.array([l1]),np.array([l2]))))
    y1, y2 = 0, 0
    win1 = vis.line(X=np.column_stack((np.array([x]),np.array([x]))), Y=np.column_stack((np.array([y1]),np.array([y2])))) 
    
    for epoch in range(MAX_EPOCH):
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string(str(datetime.now()))
        #scheduler.step()
        running_loss = 0.0
        total = 0
        correct = 0
        sys.stdout.flush()
        log_string('---train---') 
        for i,data in enumerate(trainloader, 0):
            inputs, labels = data
            b = inputs.size()[0]
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            optimizer.zero_grad()
            outputs, end_points = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            labels = labels.long()
            labels = labels.view(b)
            classify_loss = F.nll_loss(outputs, labels)
            '''         
            transform = end_points['transform']
            transform = transform.view(BATCH_SIZE, 64, 64)
            trans1 = transform.transpose(2,1)
            mat_result = torch.matmul(transform, trans1)
            eye = torch.Tensor(np.eye(64)).cuda().float()
            reg_loss = mat_result.numpy() - eye 
            reg_loss = torch.Tensor(reg_loss).cuda().float()
            reg_loss = MSEcri(reg_loss, 0)/2 #might be a problem
            ''' 
            loss = classify_loss 
            # + reg_loss
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()#loss.data[0]
            total += labels.size(0)
            predicted =Variable(predicted)
            correct += (predicted == labels).cpu().sum()
            #pred = predicted.max(1, keeddim=True)[1]
            #correct += pred.eq(labels.view_as(pred)).cpu().sum().item()
        l1 = running_loss/ float(total)
        y1 = float(correct.item())/float(total)
        #vis.line(X=np.array([epoch]), Y=np.array([l]), win=win,update='append') 
        log_string('mean loss: %f' % (running_loss / float(total)))
        log_string('accuracy: %f' % (float(correct.item()) / float(total)))
        total = 0
        correct = 0
        running_loss = 0.0
        log_string(str(datetime.now()))
        log_string('---test---')
        for j,Tdata in enumerate(testloader, 0):
            tinputs, tlabels = Tdata
            tb = tinputs.size()[0]
            tinputs, tlabels = Variable(tinputs).cuda(), Variable(tlabels).cuda()
            
            tlabels = tlabels.long().view(tb)
            toutputs, _= net(tinputs, is_training=False)
            loss = F.nll_loss(toutputs, tlabels)
            _, tpredicted = torch.max(toutputs.data, 1)
            total += tlabels.size(0)
            tpredicted = Variable(tpredicted)
            correct += (tpredicted == tlabels).cpu().sum()
            running_loss += loss.item()
            #if (j == 10): print(tpredicted, tlabels)
        l2 = running_loss/ float(total)
        y2 = float(correct.item())/float(total)
        log_string('eval mean loss: %f' % (running_loss/float(total)))
        log_string('eval accuracy: %f'% (float(correct.item()) / float(total)))   
        
        vis.line(X=np.column_stack((np.array([epoch]),np.array([epoch]))), Y=np.column_stack((np.array([l1]),np.array([l2]))), win=win, update='append')
    
        vis.line(X=np.column_stack((np.array([epoch]),np.array([epoch]))), Y=np.column_stack((np.array([y1]),np.array([y2]))), win=win1, update='append') 
    
        
        #print(tpredicted.size())
        #print(tlabels.size())
        #print(i)
        #print(j)
        #print(tpredicted, tlabels)
        #np.savetxt('tpre1.txt', np.array(tpredicted.data.view(1, tpredicted.size()[0])), fmt='%s')
        #np.savetxt('tlable1.txt', np.array(tlabels.data.view(1, tlabels.size()[0])), fmt='%s')
    torch.save(net, LOG_DIR+'/net_trans3D_transpose_10_lr.pkl')    
    #np.savetxt('tpre.txt', np.array(tpredicted.view(32)), fmt='%s')
    #np.savetxt('tlabel.txt', np.array(tlabels.view(32)), fmt='%s')
    print(predicted)
    print(labels)
    #print(tpredicted)
    #print(tlabels)
    
if __name__=='__main__':
    main()



