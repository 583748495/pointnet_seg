#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 15:31:45 2018

@author: fred
"""
import os


import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class Dataset(Dataset):
    #npoints remain a problem, data provided by navinfo doesn't have a fixed size
    def __init__(self, root, batch_size = 32, npoints = 1024, kind = 'train', normalize = 'True', normal_channel = False, modelnet10 = False, cache_size = 15000, shuffle = None):
        self.root = root
        self.batch_size = batch_size
        self.npoints = npoints
        self.normalize = normalize
        self.normal_channel = normal_channel
        self.cache_size = cache_size
        print('modelnet10:', modelnet10)
        if modelnet10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        print('category:', len(self.cat))
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        print('classes:', self.classes)
        
        shape_ids = {}
        if modelnet10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        assert(kind == 'train' or kind == 'test')
        #print('shape_ids:',len(shape_ids['train']), shape_ids['train'][0], shape_ids['train'][1])
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[kind]]
        #list of (shape_name, shape_txt_file_path) tuple
        #print('shape_names:', len(shape_names))
        #print(kind)
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[kind][i])+'.txt') for i in range(len(shape_ids[kind]))]
        #print('datapath:', len(self.datapath), self.datapath[0][1], self.datapath[1][1])
        self.cache = {}
        if shuffle is None:
            if kind == 'train':
                self.shuffle = True
            else:
                self.shuffle = False
        else:
            self.shuffle = shuffle
        
        self.idxs = np.arange(0, len(self.datapath))
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.num_batches = (len(self.datapath)+self.batch_size-1) // self.batch_size
        self.batch_idx = 0
        
    def __getitem__(self, idx):
        #if idx in self.cache:
        #    point_set, cls = self.cache[idx]
        #else:
        #print(idx, len(self.datapath))
        fn = self.datapath[idx]
        cls = self.classes[self.datapath[idx][0]]
        cls = np.array([cls]).astype(np.int32)
        point_set = np.loadtxt(fn[1], delimiter = ',').astype(np.float32)
        #take the first npoints
        point_set = point_set[0:self.npoints,:]
        if self.normalize:
            point_set = pc_normalize(point_set)
        if not self.normal_channel:
            point_set = point_set[:,0:3]
            # how does cache work???
            '''if len(self.cache) < self.cache_size:
                self.cache[idx] = (point_set, cls)'''
        #point_set = np.expand_dims(point_set, 0)
        return point_set, cls
            
    def __len__(self):
        return(len(self.datapath))

if __name__=='__main__':
    dsDir = '/home/fred/lyx/data/modelnet40_normal_resampled/'
    BATCH_SIZE = 32
    NUM_POINT = 1024
    NUM_WORKER = 4
    trainset = Dataset(dsDir, batch_size=32, npoints = NUM_POINT, shuffle = True, modelnet10=True)
    trainloader = DataLoader(trainset, batch_size=32, num_workers=NUM_WORKER)
    
    dataiter = iter(trainloader)
    inputs, labels = dataiter.next()
    print(inputs.shape)
    
    
    
    
    
    