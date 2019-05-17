# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:42:21 2019

@author: Adam
"""
from World import World
import numpy as np
from matplotlib import pyplot as plt

ratios = [[1,0,0],[0,1,0],[0,0,1],[.4,.3,.3],[0,.5,.5]]

for i in range(2):
    ratios.append(np.round(np.random.dirichlet(np.ones(3)),2))

results = []
for ratio in ratios:
    world = World(ratio,people=100)
    for i in range(1,8):
        world.each_day()
    results.append(world.get_data)    

fig, ax = plt.subplots(3,1,figsize=(10,10),sharex='all')

for i in range(len(results)):
    ax[0].plot(results[i][:,0])
    ax[1].plot(range(len(results[i])),results[i][:,1])
    ax[2].plot(range(len(results[i])),np.argmax(results[i][:,2:].astype(int),axis=1),label=str(ratios[i]))
    
plt.xlabel('Day')
plt.ylabel('Population')
plt.legend(bbox_to_anchor=(1.3, 2))
ax[0].set_title('Population')
ax[1].set_title('% Correct')
ax[2].set_title('Majority Well')
plt.show()
