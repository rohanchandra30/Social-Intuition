import json
import numpy as np

filepath = '/home/srujan/Downloads/rohan/Set_8_annotations.json'

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
        
#         print(fr_id, obj_id, x, y, dataset_id)
        
        to_list.append([fr_id, obj_id, x, y, dataset_id])

    
to_array = np.asarray(  to_list)
