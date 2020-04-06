from matplotlib import pylab, pyplot as plt
import networkx as nx
import json
import os
import numpy as np
from adjacency import *
from scipy import signal

# SET THESE THREE PARAMS
'''
num:     [0-8] See "sets" variable. 

         Set 2 has 3 agents, so you have to give options '20', '21', and '22' 
         which correspond to the 1st, 2nd, and 3rd agents respectively. 
         
         Set 7 has 2 agents, so you have to set'70' or '71' accordingly.
        
cen_num: 0:closeness (num == 0,20,21,22, 9), 1:degree (num == 1 and 6), 2: eigenvector (num == 3,4)

color:   Turn on region coloring
'''
num = 6
cen_num = 0
color = False

# ================================================================================================

agent_num = 0
if 20 <= num <= 30:
    agent_num = num % 20
    num = int(num / 10)

if 40 <= num <= 80:
    agent_num = num % 70
    num = int(num / 10)

sets = ['4',                        # 0
        '12_55_47',                 # 1
        '11_30_18',                 # 2
        '7_30_18_2',                # 3
        '10_30_18',                 # 4
        '13-2019-08-27-22-30-18',   # 5
        '16_30_18',                 # 6
        '9_47_10',                  # 7
        '5_55_47',                  # 8
        '10_55_47']                 # 9
frames = [[0, 60], [47, 80], [[55, 95], [70, 98],[70, 98]],
          [52, 98], [14, 50], [59, 85], [65, 72], [ [9, 35], [50, 98]],
          [58, 90], [10, 98]]
agent_IDs = [983, 2677, [2810, 2958, 2959], 1336, 3494, 1295, 1786, [[2562], [2564]], 1750, 868]
radius = [10, 20, 10, 20, 20, 10, 20, 10, 10, 10]
agent_labels = ['Black Car', 'White Car', 'White Car', 'White Bus',
                'White Truck', 'White Lorry', 'Motorbike', 'Scooter', 'Scooter', 'Motorbike']
centrality_labels = ['Closeness Centrality Value', 'Degree Centrality Value', 'Eigenvector Centrality Value']
thresholds = [[0, 35, 60], [47, 61, 80], [[55, 80, 95], [70, 80, 98], [70, 80, 98]], [52, 73, 98], [14, 40, 50],
              [59, 75, 85], [65, 68, 72], [ [9, 18, 35], [50, 75, 98]], [58, 80, 90], [24, 36]]

video_set = sets[num]
frame = frames[num] if (num != 2 and num != 7) else frames[num][agent_num]
agent_ID = agent_IDs[num]
rad = radius[num]
agent_label = agent_labels[num]
centrality_label = centrality_labels[cen_num]
x_lims = thresholds[num] if (num != 2 and num != 7) else thresholds[num][agent_num]

# filenames = os.listdir('data/')

# READ CSV FILE PANDAS
filepath = 'data/set' + str(video_set) + '_annotations_utm.json'

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
agent_idx = list(obj_IDs).index(agent_ID[agent_num]) if (num == 2 or num == 7) else list(obj_IDs).index(agent_ID)
degree_mat, adj_mats = generate_adjacency(to_array, rad)
weave_list = []
weave_list2 = []
weave_list3 = []
for fr, item in enumerate(adj_mats):
    G = nx.from_numpy_array(item['adj_matrix'])
    if cen_num == 0:
        g = nx.closeness_centrality(G)
    elif cen_num == 1:
        g = nx.degree_centrality(G)
    else:
        g = nx.eigenvector_centrality(G, max_iter=10000)
    # eg = nx.eigenvector_centrality(G)
    # print(fr,g[agent_idx])
    # if fr < 60:
    if frame[0] <= fr <= frame[1]:
        weave_list.append(g[0])  # num/id- 0,0, 1/1, 2/0, 3/0, 4/0
        weave_list2.append(g[agent_idx])
        # weave_list2.append(cg[7])
    # weave_list3.append(cg[8])

buffer2 = 7 if len(weave_list2) % 2 == 0 else 8
buffer = 7 if len(weave_list) % 2 == 0 else 8
if num != 6:
    weave_list2 = signal.savgol_filter(weave_list2, len(weave_list2)-buffer2, 3)
    # weave_list = signal.savgol_filter(weave_list, len(weave_list)-buffer, 3)


cmaps = ['#0038ff', '#00a4b2', '#4c6fb8', '#2c5167', '#9fd9ea']
LineThick = 5
FontSize = 40

fig, ax = plt.subplots(figsize=(11.0, 8.0))
x = np.arange(frame[0], frame[1] + 1)
y = weave_list2
# x = np.arange(frame[0], frame[1])
# y = weave_list2[2:] - 2 * weave_list2[:-1]
if len(agent_label.split()) > 1:
    agent_color = agent_label.split()[0]
else:
    agent_color = 'Gold'
ax.plot(x, y, linewidth=LineThick, label=agent_label, color='k')
if num == 0:
    ax.plot(x, weave_list, linewidth=LineThick, color='k')
    # ax.plot(range(frame[0], frame[1]+1), weave_list, linewidth=LineThick, label='Red Car', color='Tomato')
if color:
    for i in range(0, len(x_lims) - 1):
        ax.fill_between([x_lims[i], x_lims[i + 1]],
                        np.max(y[x_lims[i]-frame[0]:x_lims[i + 1]-frame[0]+1]) + 0.002,
                        np.min(y[x_lims[i]-frame[0]:x_lims[i + 1]-frame[0]+1]) - 0.002,
                        facecolor=cmaps[i], alpha=0.6, interpolate=True)
        # ax.fill_between([25, 35], np.max(y),np.min(y), facecolor='red', alpha=0.2, interpolate=True)
        # ax.fill_between([75, 85], np.max(y),np.min(y), facecolor='red', alpha=0.2, interpolate=True)
plt.grid(True)
plt.xlabel('Frame Number', fontsize=FontSize)
plt.ylabel(centrality_label.split()[0], fontsize=FontSize)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(FontSize - 7)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(FontSize - 7)
# Turn off tick labels
# ax.set_yticklabels([])
# legend = plt.legend(loc='lower left', bbox_to_anchor=(0, 1.01), ncol=2,
#                     borderaxespad=0, fontsize=FontSize - 5, fancybox=True,
#                     facecolor='green', framealpha=0.4)
# frame = legend.get_frame()
# frame.set_facecolor('green')
# frame.set_edgecolor('red')
# plt.savefig('images/' + video_set + '_' + agent_label + '.png', bbox_inches='tight')
# fig2, ax2 = plt.subplots(figsize=(15.0, 12.0))
# line, = ax2.plot(x, y,linewidth=LineThick, color='k')
# plt.grid(True)
# # plt.xlabel('Frame Number', fontsize=FontSize)
# # plt.ylabel(centrality_label.split()[0], fontsize=FontSize)
# for tick in ax2.xaxis.get_major_ticks():
#     tick.label.set_fontsize(FontSize - 7)
# for tick in ax2.yaxis.get_major_ticks():
#     tick.label.set_fontsize(FontSize - 7)
# y_init = list([0]*len(y))



# for n in range(len(x)):
#     line.set_data(x[:n], y[:n])
#     # ax2.axis([0, 100, 0, ])
#     fig.canvas.draw()
#     plt.savefig('video_materials/' + video_set + '_' + '{}.png'.format(n), bbox_inches='tight')


# plt.plot(range(frame[0], frame[1]+1), weave_list2, linewidth= LineThick, label=agent_label )
# if num==0:
#     plt.plot(range(frame[0], frame[1]+1), weave_list, linewidth= LineThick,label="Scooter" )
# plt.grid(True)
# plt.xlabel('Time (Frame Number)', fontsize=FontSize)
# plt.ylabel(centrality_label, fontsize=FontSize)
# plt.legend(loc='upper left', fontsize=FontSize)
plt.show()
