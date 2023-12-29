import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import nn
from graph_rl import *
from rl_agent import *
import pickle
import random

np.random.seed(2022)
random.seed(2022)
torch.manual_seed(2022)

device = torch.device('cuda:0')
lu = './'
chu = './'
gpu = True

ming=['chord','koorde','kadem','leet','p2p', 'c2']
ceshi=['Bamboo', 'Broose', 'Chord', 'Gia', 'Kademlia', 'Koorde', 'Nice']

# The training set
train = []

i1 = ming[0]
if not os.path.exists(chu + i1 + '/'):
    os.mkdir(chu + i1 + '/')
print('#'*6, i1, '#'*6)
for i2 in range(7):
    model = Agent()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    print('+'*6, ceshi[i2], '+'*6)
    w2 = lu + i1 + '/' + ceshi[i2] + '/'
    c2 = chu + i1 + '/' + ceshi[i2] + '/'
    if not os.path.exists(c2):
        os.mkdir(c2)
    yuzhi = 0.999
    for epoch in range(1):
        print('Epoch ==',str(epoch),'==')
        for number in range(len(train)):
            du = w2 + train[number] + '.pkl'
            env = pickle.load(open(du, 'rb'))
            memory = []
            rewards = 0
            idx = 0
            cishu = 0
            print('Train == Graph ',str(number),' ==')
            bai, baiz, hei, heiz = env.metrics()
            while True:
                tags = torch.from_numpy(env.node_features[:,0:1]).float().cuda() * 10000000000000000
                state = torch.from_numpy(env.node_features[:,:6]).float().cuda()
                probs = model(state, tags)
                m = Categorical(probs[:,0])
                action = m.sample()
                gailv = m.log_prob(action)
                if env.node_features[action.item(),0] == 1:
                    print('--  detect  --',str(action.item()),'--  Already checked !!!')
                    continue
                idx += 1
                reward, done, tag = env.patrol_node(action.item())
                rewards += reward
                memory.append([gailv, reward])
                if tag == 1:
                    bai += 1
                if tag == 2:
                    hei += 1
                if idx % 10 == 0 or done:
                    if rewards < 0.01:
                        cishu += 1
                    else:
                        cishu = 0
                    if cishu >= 10 and ((bai+hei) / (baiz+heiz)) > yuzhi:
                        done = True
                    V = 0
                    VS = []
                    for _, r in memory[::-1]:
                        V = V * 0.9 + r
                        VS.insert(0, V)
                    VS = torch.tensor(VS)
                    VS = (VS - VS.mean()) / (VS.std() + torch.finfo(torch.float).tiny)#
                    loss = 0
                    for V, (log_prob, _) in zip(VS, memory):
                        loss += -V * log_prob
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    memory = []
                    rewards = 0
                if done:
                    print('Done~~~~')
                    break
            torch.save(model, c2 + 'model_' + str(epoch) + '_' + str(shun.index(number)) + '.pt')
