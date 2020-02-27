from matplotlib import pylab, pyplot as plt
import networkx as nx
import json, os
import numpy as np
from adjacency import *

# SET THESE TWO PARAMS
num = 4 # [0-6] See "sets" variable. Set 2 has 3 agents, so you have to give options '20', '21', and '22' which correspond to the 1st, 2nd, and 3rd agents respectively
cen_num = 0 #0:closeness (num == 0,20,21,22), 1:degree (num == 1 and 6), 2:eigenvector (num == 3,4)
if num >= 20:
    agent_num = num%20
    num = int(num/10)


centrality_labels = ['Closeness Centrality Value','Degree Centrality Value','Eigenvector Centrality Value']
LineThick = 3
FontSize = 24
sets = ['4',        #0
        '12_55_47', #1
        '11_30_18', #2
        '7_30_18_2',#3
        '10_30_18', #4
        '13-2019-08-27-22-30-18', #5
        '16_30_18', #6
        '10']       #7 (not operational at the moment)
frames= [[0,60],[47,80],[[55,95],[70,98]], [52,98],[14,50], [59,85], [65,72]]
agent_IDs = [983,2677,[2810,2958,2959],1336,3494,1295,1786]
radius = [10,20,10,20,20,10,10]
agent_labels = ['Black Car', 'White Car', 'White Car', 'White Bus', 'White Truck', 'White Lorry', 'Motorbike']
centrality_label = centrality_labels[cen_num]

set = sets[num]
frame = frames[num] if num != 2 else frames[num][agent_num]
agent_ID = agent_IDs[num]
rad = radius[num]
agent_label = agent_labels[num]
# filenames = os.listdir('data/')
filepath = 'data/set'+str(set)+'_annotations_utm.json'

with open(filepath) as json_file:
    sample_annotation = json.load(json_file)

dataset_id = 1

to_list = []

for each_frame in sample_annotation:

    fr_id = each_frame['id']
    each_annotation = each_frame['annotations']
    for each_object in each_annotation:
        obj_id = each_object['classId']
        x = each_object['geometry']['position']['x']
        y = each_object['geometry']['position']['y']
        to_list.append([fr_id, obj_id, x, y, dataset_id])

to_array = np.asarray(to_list)
obj_IDs = np.unique(to_array[:, 1]).astype(int)
agent_idx = list(obj_IDs).index(agent_ID[agent_num]) if num==2 else list(obj_IDs).index(agent_ID)
adj_mats = generate_adjacency(to_array,rad)
weave_list = []
weave_list2 = []
weave_list3 = []
for fr,item in enumerate(adj_mats):
    G = nx.from_numpy_array(item['adj_matrix'])
    if cen_num == 0:
        g=nx.closeness_centrality(G)
    elif cen_num == 1:
        g = nx.degree_centrality(G)
    else:
        g = nx.eigenvector_centrality(G, max_iter=10000)
    # eg = nx.eigenvector_centrality(G)
    print(fr,g[agent_idx])
    if fr >=frame[0] and fr <= frame[1]:
    # if fr < 60:
        weave_list.append(g[0]) # num/id- 0,0, 1/1, 2/0, 3/0, 4/0
        weave_list2.append(g[agent_idx])
        # weave_list2.append(cg[7])
    # weave_list3.append(cg[8])

from scipy import signal

buffer2 = 7 if len(weave_list2)%2==0 else 8
buffer = 7 if len(weave_list)%2==0 else 8
if num != 6:
    weave_list2 = signal.savgol_filter(weave_list2, len(weave_list2)-buffer2, 3)
    # weave_list = signal.savgol_filter(weave_list, len(weave_list)-buffer, 3)
plt.plot(range(frame[0], frame[1]+1), weave_list2, linewidth= LineThick, label=agent_label )
if num==0:
    plt.plot(range(frame[0], frame[1]+1), weave_list, linewidth= LineThick,label="Scooter" )
plt.grid(True)
plt.xlabel('Time (Frame Number)', fontsize=FontSize)
plt.ylabel(centrality_label, fontsize=FontSize)
plt.legend(loc='upper left', fontsize=FontSize)
plt.show()
