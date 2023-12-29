import os
import h5py
import numpy as np
import random

class RLGraph(object):
    def __init__(self, labels=None, edge_array=None, node_features=None):
        self.labels = labels
        self.edge_array = edge_array
        self.node_features = node_features #flags, pred, degree, d_benign, d_attack, d_none, (traffic_size)
        self.neighbors = []

    def edge_traverse(self):
        self.node_features = np.zeros((self.labels.shape[0],7)).astype(np.int32) #7 if traffic_size
        self.neighbors = [[] for i in range(self.labels.shape[0])]
        for i1 in self.edge_array:
            self.neighbors[i1[0]].append(i1[1])
            self.neighbors[i1[1]].append(i1[0])
        for i2 in range(self.labels.shape[0]):
            self.neighbors[i2] = list(set(self.neighbors[i2]))
            self.node_features[i2,2] = len(self.neighbors[i2])
            self.node_features[i2,5] = len(self.neighbors[i2])
        self.edge_array=0
    
    def end_detection(self):
        if self.node_features[:,0].sum() == self.node_features.shape[0]:
            return True
        #end_conditions
        elif ((self.node_features[:,1] <= 10000).sum() + (self.node_features[:,1] >= 90000).sum()) == self.node_features.shape[0]:
            return True
        else:
            return False

    def metrics(self):
        bai = (self.node_features[self.labels[:] == 0,1] < 50000).sum()
        hei = (self.node_features[self.labels[:] == 1,1] >= 50000).sum()
        baiz = (self.labels[:] == 0).sum()
        heiz = (self.labels[:] == 1).sum()
        return bai, baiz, hei, heiz

    def patrol_node(self, node_id):
        self.node_features[node_id,0] = 1
        liez = self.neighbors[node_id]
        reward = 0.0001
        tag = 0
        if self.labels[node_id] == 0:
            if self.node_features[node_id,1] >= 50000:
                reward += abs(float(self.node_features[node_id,1] / 100000.) - 0.) * 10
                tag = 1
            for i1 in liez:
                self.node_features[i1,3] += 1
                self.node_features[i1,5] -= 1
                if self.node_features[i1,0] == 1:
                    if self.labels[i1] == 1 and (self.node_features[node_id,1] >= 50000):
                        reward += 0.5 * 10
                    continue
            self.node_features[node_id,1] = 0

        if self.labels[node_id] == 1:
            if self.node_features[node_id,1] < 50000:
                reward += abs(float(self.node_features[node_id,1] / 100000.) - 1.) * 10
                tag = 2
            for i1 in liez:
                self.node_features[i1,4] += 1
                self.node_features[i1,5] -= 1
                if self.node_features[i1,0] == 1:
                    if self.labels[i1] == 0 and self.node_features[node_id,1] < 50000:
                        reward += 0.5 * 10
                    continue
            self.node_features[node_id,1] = 100000
        done = False
        if self.node_features[:,0].sum() == self.node_features.shape[0]:
            done = True
        return max(reward, 0.0001), done, tag
