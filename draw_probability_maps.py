import csv
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

from adjacency import *
from scipy import signal


num = 13 # 11-16
cen_num = 1
color = False

start_frames = []
end_frames = []
with open('aggressive_frames.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            video_names = row
            num_videos = len(video_names)
        else:
            starts = np.empty(num_videos, dtype=float)
            ends = np.empty(num_videos, dtype=float)
            for r, word in enumerate(row):
                try:
                    starts[r] = float(word.split()[0])
                    ends[r] = float(word.split()[-1])
                except ValueError:
                    starts[r] = -np.inf
                    ends[r] = np.inf
            start_frames.append(starts)
            end_frames.append(ends)
        line_count += 1
    start_frames = np.stack(start_frames)
    end_frames = np.stack(end_frames)

# ================================================================================================

# 0 - '30_18_7(front)' -    SLC         77.0    79  2,
# 1 - '30_18_10(rear)' -    SLC         28.125  31  2.875,
# 2 - '30_18_11(rear)' -    OT          94.8125 96  1.1875,
# 3 - '30_18_11(front)' -   W           67.375  69  1.625,
# 4 - '30_18_13(front)' -   W           71.5    75  3.5,
# 5 - '30_18_16(front)' -   OS          70.0625 68  2.0625,
# 6 - '47_10_9(front)' -    OS          17.875  18  0.125,
# 7 - '47_10_9(rear)' -     OT          75.0625 75  0.0625,
# 8 - '55_47_4(rear)' -     OT          45.0    49  4,
# 9 - '55_47_5(rear)' -     SLC         72.5625 76  3.4375,
# 10 - '55_47_12(front) -   OS          65.125
# 11 - 1001.csv         -   SLC         31.5625 34  2.4375
# 12 - 1351.csv         -   OS          27.625  25  2.625
# 13 - 1901.csv         -   OS          32.1875 31  1.1875
# 14 - 3451.csv         -   OS          25.875  28  2.125
# 15 - 4351.csv         -   OS          32.75   30  2.75
# 16 - 4751.csv         -   OS          32.375  29  3.375

agent_num = 0
if 20 <= num <= 30:
    agent_num = num % 20
    num = int(num / 10)

if 40 <= num <= 80:
    agent_num = num % 70
    num = int(num / 10)

# frames = [[0, 60],
#           [47, 80],
#           [[55, 95], [70, 98]],
#           [52, 98],
#           [14, 50],
#           [59, 85],
#           [65, 72],
#           [[50, 98], [9, 35]],
#           [58, 90],
#           [10, 98]]
frames = [[52, 98], [14, 50], [70, 98], [55, 95], [59, 85], [65, 72],
          [9, 35], [50, 98], [0, 60], [58, 90], [10, 98],
          [12, 48], [10, 50], [12, 50], [9, 43], [11, 50], [10, 50]]
# agent_IDs = [983, 2677, [2810, 2958, 2959], 1336, 3494, 1295, 1786, [[2562], [2564]], 1750, 868]
agent_IDs = [1336, 3494, 2959, 2810, 1295, 1786, 2562, 2564, 983, 1750, 2677,
             33138, 102040, 32227, 29137, 44854, 28362]
radius = [10, 20, 10, 10, 20, 10, 10, 10, 10, 10, 10,
          10, 10, 16, 10, 20, 50]
agent_labels = ['Black Car', 'White Car', 'White Car', 'White Bus',
                'White Truck', 'White Lorry', 'Motorbike', 'Scooter', 'Scooter', 'Motorbike', 'White Car',
                'Red Agent', 'Red Agent', 'Red Agent', 'Red Agent', 'Red Agent', 'Red Agent']
centrality_labels = ['Closeness Centrality Value', 'Degree Centrality Value', 'Eigenvector Centrality Value']
# thresholds = [[0, 35, 60], [47, 61, 80], [[55, 80, 95], [70, 80, 98], [70, 80, 98]], [52, 73, 98], [14, 40, 50],
#               [59, 75, 85], [65, 68, 72], [[50, 75, 98], [9, 18, 35]], [58, 80, 90], [21, 26]]

# frame = frames[num] if (num != 2 and num != 7) else frames[num][agent_num]
frame = frames[num]
agent_ID = agent_IDs[num]
rad = radius[num]
agent_label = agent_labels[num]
centrality_label = centrality_labels[cen_num]
# x_lims = thresholds[num] if (num != 2 and num != 7) else thresholds[num][agent_num]
mean_start_frame = np.mean(start_frames, axis=0)
std_start_frame = np.std(start_frames, axis=0)
mean_end_frame = np.mean(end_frames, axis=0)
std_end_frame = np.std(end_frames, axis=0)
x_lims = [mean_start_frame[num], mean_end_frame[num]]
x_start_lims = [mean_start_frame[num] - std_start_frame[num],
                mean_start_frame[num] + std_start_frame[num]]
x_end_lims = [mean_end_frame[num] - std_end_frame[num],
              mean_end_frame[num] + std_end_frame[num]]

for v in range(len(video_names)):
    if video_names[v].split('_')[-2].lower() == 'gt':
        video_names[v] = '_'.join(video_names[v].split('_')[:-2])
    else:
        video_names[v] = '_'.join(video_names[v].split('_')[:-1])
# filenames = os.listdir('data/')

dataset_id = 1
to_list = []
if num < 11:
    file_path = os.path.join('data', video_names[num] + '.json')
    with open(file_path) as json_file:
        sample_annotation = json.load(json_file)

    for each_frame in sample_annotation:
        fr_id = each_frame['id']
        each_annotation = each_frame['annotations']
        for each_object in each_annotation:
            obj_id = each_object['classId']
            x = each_object['geometry']['position']['x']
            y = each_object['geometry']['position']['y']
            to_list.append([fr_id, obj_id, x, y, dataset_id])
else:
    file_path = os.path.join('argo_data', video_names[num] + '.csv')
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        fr = 0
        for line_count, row in enumerate(csv_reader):
            if line_count == 0:
                continue
            else:
                if line_count == 1:
                    curr_frame = float(row[0])
                    prev_frame = float(row[0])
                else:
                    prev_frame = np.copy(curr_frame)
                    curr_frame = float(row[0])
                    if curr_frame != prev_frame:
                        fr += 1
                obj_id = int(row[1].split('-')[-1])
                x = float(row[3])
                y = float(row[4])
                to_list.append([fr, obj_id, x, y, dataset_id])

to_array = np.asarray(to_list)
obj_IDs = np.unique(to_array[:, 1]).astype(int)
# agent_idx = list(obj_IDs).index(agent_ID[agent_num]) if (num == 2 or num == 7) else list(obj_IDs).index(agent_ID)
agent_idx = list(obj_IDs).index(agent_ID)
adj_mats = generate_adjacency(to_array, rad)
weave_list = []
weave_list2 = []
weave_list3 = []
fr_list = []
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
        fr_list.append(fr)
        # weave_list2.append(cg[7])
    # weave_list3.append(cg[8])

buffer2 = 7 if len(weave_list2) % 2 == 0 else 8
buffer = 7 if len(weave_list) % 2 == 0 else 8
if num != 5:
    weave_list2 = signal.savgol_filter(weave_list2, len(weave_list2)-buffer2, 3)
    # weave_list = signal.savgol_filter(weave_list, len(weave_list)-buffer, 3)
else:
    weave_list2 = np.asarray(weave_list2)


cmaps = ['#0038ff', '#00a4b2', '#4c6fb8', '#2c5167', '#9fd9ea']
LineThick = 5
FontSize = 40

fig, ax = plt.subplots(figsize=(11.0, 8.0))
if num in [12, 13, 15, 16]:
    x = np.arange(frame[0], frame[1] - 1)
else:
    x = np.arange(frame[0], frame[1] + 1)
y = weave_list2
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
                        np.max(
                            y[int(np.floor(x_lims[i])) - frame[0]:int(np.ceil(x_lims[i + 1])) - frame[0] + 1]) + 0.002,
                        np.min(
                            y[int(np.floor(x_lims[i])) - frame[0]:int(np.ceil(x_lims[i + 1])) - frame[0] + 1]) - 0.002,
                        facecolor=cmaps[i], alpha=0.3, interpolate=True)
        ax.fill_between([x_start_lims[i], x_start_lims[i + 1]],
                        np.max(
                            y[int(np.floor(x_lims[i])) - frame[0]:int(np.ceil(x_lims[i + 1])) - frame[0] + 1]) + 0.002,
                        np.min(
                            y[int(np.floor(x_lims[i])) - frame[0]:int(np.ceil(x_lims[i + 1])) - frame[0] + 1]) - 0.002,
                        facecolor=cmaps[i + 1], alpha=0.3, interpolate=True)
        ax.fill_between([x_end_lims[i], x_end_lims[i + 1]],
                        np.max(
                            y[int(np.floor(x_lims[i])) - frame[0]:int(np.ceil(x_lims[i + 1])) - frame[0] + 1]) + 0.002,
                        np.min(
                            y[int(np.floor(x_lims[i])) - frame[0]:int(np.ceil(x_lims[i + 1])) - frame[0] + 1]) - 0.002,
                        facecolor=cmaps[i + 1], alpha=0.3, interpolate=True)
        # ax.fill_between([25, 35], np.max(y),np.min(y), facecolor='red', alpha=0.2, interpolate=True)
        # ax.fill_between([75, 85], np.max(y),np.min(y), facecolor='red', alpha=0.2, interpolate=True)
    # plt.errorbar(start_frames[num],
    #              np.mean(y[int(np.floor(x_lims[i])) - frame[0]:int(np.ceil(x_lims[i + 1])) - frame[0] + 1]),
    #              x_err=start_frames[:, num] - mean_start_frame[num])
# ax.plot([31.5625 for _ in range(2)], [np.min(y) - 0.002, np.max(y) + 0.002], linestyle='--', linewidth=LineThick)
# ax.plot([34 for _ in range(2)], [np.min(y) - 0.002, np.max(y) + 0.002], linestyle='-.', linewidth=LineThick)
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

# plt.plot(range(frame[0], frame[1]+1), weave_list2, linewidth= LineThick, label=agent_label )
# if num==0:
#     plt.plot(range(frame[0], frame[1]+1), weave_list, linewidth= LineThick,label="Scooter" )
# plt.grid(True)
# plt.xlabel('Time (Frame Number)', fontsize=FontSize)
# plt.ylabel(centrality_label, fontsize=FontSize)
# plt.legend(loc='upper left', fontsize=FontSize)
fig2, ax2 = plt.subplots(figsize=(15.0, 12.0))
line, = ax2.plot(x, y,linewidth=LineThick, color='k')
plt.grid(True)

for tick in ax2.xaxis.get_major_ticks():
    tick.label.set_fontsize(FontSize - 7)
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(FontSize - 7)
# y_init = list([0]*len(y))

#
for n in range(len(x)):
    line.set_data(x[:n], y[:n])
    # ax2.axis([0, 100, 0, ])
    fig.canvas.draw()
    plt.savefig('video_materials/' + video_names[num] + '_' + '{}.png'.format(n), bbox_inches='tight')




plt.show()
# plt.savefig('images/' + video_names[num] + '_' + agent_label + '.png', bbox_inches='tight')

weave_list_grad = np.abs(weave_list2[1:] - weave_list2[:-1])
estimated_mean_frame = fr_list[np.argmax(weave_list_grad[1:]) + 1]
data_mean_frame = np.mean(x_lims)
temp = 1
