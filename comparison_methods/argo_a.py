from math import sqrt , pow
import heapq, json
from operator import itemgetter
import pickle
import scipy.io
import pandas as pd
from math import pow
import time
from sklearn import preprocessing
import os
from collections import Counter
import numpy as np
from sklearn.linear_model import LinearRegression



# data = np.load('argo_data.npy', allow_pickle=True)
# data = data[0]


# Format Singapore Dataset

filepath = '../Spectral-Intuition/data/set4_annotations_utm.json'
with open(filepath) as json_file:
	data = json.load(json_file)

SgData = []

for each_frame in data:
	frame = []
	fr_id = each_frame['id']
	each_annotation = each_frame['annotations']
	for each_object in each_annotation:
		obj_id = each_object['classId']
		x = each_object['geometry']['position']['x']
		y = each_object['geometry']['position']['y']
		frame.append([obj_id,[x,y]])
	SgData.append(frame)

data = SgData
############################ Avg Velocity ######################
def computeDist(x1,y1,x2,y2):
	return sqrt(pow(x1-x2,2) + pow(y1-y2,2))

vel = [] 
dict_items = {}
for i in range(len(data)):
	for j in data[i]:
		if j[0] not in dict_items:
			dict_items[j[0]] = [j[1][0], j[1][1], j[1][0], j[1][1], i, i]
		else:
			values = dict_items[j[0]]
			values[2] = j[1][0]
			values[3] = j[1][1]
			values[5] = i

vel_dict = {}
for k in dict_items.keys():
	vel_dict[k] = (computeDist(dict_items[k][0], dict_items[k][1], dict_items[k][2], dict_items[k][3]))/(dict_items[k][5]-dict_items[k][4] + 0.001)

#print(vel_dict)
vel.append(dict_items)
file = open('positions', 'wb')
pickle.dump(vel, file)
file.close()



###################### Relative Neighbor Velocity #################3
def computeKNN(curr_dict, ID, k):
	dists = {}
	for i in curr_dict:
		if i[0]==ID:
			ID_x = i[1][0]
			ID_y = i[1][1]

	for i in curr_dict:
		if i[0]!=ID:
			dists[i[0]] = computeDist(i[1][0], i[1][1], ID_x,ID_y)
	KNN_IDs = dict(heapq.nlargest(k,dists.items(),key=itemgetter(1)))
	return list(KNN_IDs.keys())


neighbors = []
dict_items = {}
for i in range(len(data)):
	for j in data[i]:
		if j[0] not in dict_items:
			dict_items[j[0]] = computeKNN(data[i],j[0],4)
		else:
			values = computeKNN(data[i],j[0],4)
			for v in values:
				if v not in dict_items[j[0]]:
					dict_items[j[0]].append(v)

neighbour_vel = {}
for k in dict_items.keys():
	values = dict_items[k]
	sum = 0
	for i in values:
		sum += vel_dict[i]
	avg = sum/len(values)
	neighbour_vel[k] = avg

#print(neighbour_vel)


########################## Distance ########################################

def closest_car(curr_dict,ID,k):
	dists = {}
	for i in curr_dict:
		if i[0] == ID:
			ID_x = i[1][0]
			ID_y = i[1][1]

	for i in curr_dict:
		if i[0] != ID:
			dists[i[0]] = computeDist(i[1][0], i[1][1], ID_x, ID_y)
	try:
		least = min(dists)
	except:
		pass
	return [least]

closest = []
dict_items = {}
for i in range(len(data)):
	print(i)
	for j in data[i]:
		if len(data[i]) >1:
			if j[0] not in dict_items:
				dict_items[j[0]] = closest_car(data[i],j[0],4)
			else:
				values = closest_car(data[i],j[0],4)
				for v in values:
					if v not in dict_items[j[0]]:
						dict_items[j[0]].append(v)

close_dist = {}
for k in dict_items.keys():
	values = dict_items[k]
	sum = 0
	for i in values:
		sum += i
	avg = sum/len(values)
	close_dist[k] = avg

#print(close_dist)


labels = np.load('argo_labels.npy', allow_pickle=True)
labels = labels[0]



x = []
for i in neighbour_vel.keys():
	a =  [neighbour_vel[i],close_dist[i]]
	x.append(a)
	# x[i].append(close_dist[i])
use = [0] * len(x)


train_y = np.array(use[:10])
test_y = np.array(use[5:])
train_x = np.array(x[:10])
test_x = np.array(x[5:])

reg = LinearRegression().fit(train_x, train_y)
print(reg)
predict = reg.predict(test_x)

