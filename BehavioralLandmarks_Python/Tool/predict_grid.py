# IMPORTS
from gc import isenabled
from importlib.metadata import requires
import os
from turtle import right, shape
from venv import create
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from sklearn.model_selection import train_test_split
from keras.models import load_model
import tifffile
import json
from sklearn.metrics import confusion_matrix
import glob
from scipy.ndimage import zoom

# cd C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Tool

def apply_temporal_classifier(output_dict):
    folder_path = 'C:\\PROJECTS\\BehavioralLandmarks\\BehavioralLandmarks_Python\\Tool\\Data\\TemporalLandmarks'
    all_files = os.listdir(folder_path)
    tif_files = [file for file in all_files if file.lower().endswith('.tif')]
    loaded_images = []
    index = 0
    my_dict={}
    behavior_dict={}
    behavior_dict['0']='Approach'
    behavior_dict['1']='Wander'
    behavior_dict['2']='CircleAround'
    behavior_dict['3']='Avoid'
    for tif_file in tqdm(tif_files):
        image_path = os.path.join(folder_path, tif_file)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        loaded_images.append(image)
        name = (tif_file.split("tif_")[1]).split(".")[0]
        my_dict[name] = 'None'

    x = np.array(loaded_images)
    outputs = classifier_T(x)
    outputs2 = identifier(x)
    prediction = np.argmax(outputs,axis=1)
    prediction2 = np.argmax(outputs2,axis=1)
    k=0
    for key, value in my_dict.items():
        T = prediction[k]+1
        my_dict[key]=str(T)
        output_dict[key]="T_"+str(T) #+"_"+behavior_dict[str(prediction2[k])]
        k += 1
    with open('temporal.json', 'w') as json_file:
        json.dump(my_dict, json_file)

    return output_dict

def tackle_rest(output_dict,det_dict):
    for key,value in det_dict.items():
        name = key.split("tif_")[1]
        output_dict[name]=value
    return output_dict

def create_env():
    folder_path = "C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Tool\Data"
    all_files = os.listdir(folder_path)
    tif_files = [file for file in all_files if file.lower().endswith('.tif')]
    for tif_file in tif_files:
        image_path = os.path.join(folder_path, tif_file)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        shape_0 = image.shape[0]
        shape_1 = image.shape[1]

    with open('grid.json', 'w') as json_file:
        json.dump(output_dict, json_file)
    behavior_dict={}
    behavior_dict['Approach'] = np.array([0,0,1])
    behavior_dict['Wander'] = np.array([0,1,0])
    behavior_dict['CircleAround'] = np.array([1,0,0])
    behavior_dict['Avoid'] = np.array([1,0,1])
    behavior_dict['BI'] = np.array([1,1,1])

    # Point List
    points = []

    # Create environment:
    res_x = dim_x * 1
    res_y = dim_y * 1
    ratio = int(res_x/res_y)
    if (ratio > 0):
        env_x = shape_1 
        env_y = shape_0 
    else:
        env_y = shape_0 
        env_x = shape_1 
    ratio_r = int(env_y/n_rows)
    ratio_c = int(env_x/n_cols)
    env = np.zeros((env_y,env_x,3),dtype = np.float32)
    image_full = image
    id = 0
    for key, value in output_dict.items():
        col = int(key[0])
        row = int(key[1])
        centre_r = int((row*ratio_r+(row+1)*ratio_r)/2)
        centre_c = int((col*ratio_c+(col+1)*ratio_c)/2)
        env[row*ratio_r:(row+1)*ratio_r,col*ratio_c:(col+1)*ratio_c,:] = np.array([0.9,0.9,0.9])
        left = row*ratio_r
        right = (row+1)*ratio_r
        top = col*ratio_c
        bottom = (col+1)*ratio_c
        w = 15
        values = value.split("_")
        if value == "SpatialLandmarks":
            # Spatial Landmark
            env[centre_r-w:centre_r+w,centre_c-w:centre_c+w,:] = np.array([0,0,1])
            image_full[centre_r-w:centre_r+w,centre_c-w:centre_c+w,:3] = np.array([0,0,1])*255
            points.append(GraphPoint(id, centre_r,centre_c,1,int(key[0]),int(key[1])))
        elif value == "NoLandmarks":
            # No Landmark
            env[centre_r-w:centre_r+w,centre_c-w:centre_c+w,:] = np.array([1,0,0])
            points.append(GraphPoint(id, centre_r,centre_c,0,int(key[0]),int(key[1])))
        elif value == "BI":
            env[centre_r-w:centre_r+w,centre_c-w:centre_c+w,:] = np.array([0,0,0])
            points.append(GraphPoint(id, centre_r,centre_c,2,int(key[0]),int(key[1])))
        else:
            # Temporal
            env[centre_r-w:centre_r+w,centre_c-w:centre_c+w,:] = np.array([70,145,40])/255
            image_full[centre_r-w:centre_r+w,centre_c-w:centre_c+w,:3] = np.array([70,145,40])
            points.append(GraphPoint(id, centre_r,centre_c,0,int(key[0]),int(key[1])))
            print("Changes at time: T = ",values[1])
        id += 1
    tifffile.imwrite("test.tif",env)
    tifffile.imwrite("testFull.tif",image_full)
    return points , env

class GraphPoint:
    def __init__(self, id, row, col, isEnabled, gridID_1, gridID_2):
        self.id = id
        self.row = row
        self.col = col
        self.isEnabled = isEnabled
        self.isVisited = False 
        self.gridID_1 = gridID_1
        self.gridID_2 = gridID_2

    def isNeighbor(self,point):
        gridID_1,gridID_2 = point.get_gridID()
        if (abs(gridID_1-self.gridID_1) <= 1):
            if (abs(gridID_2-self.gridID_2) <= 1):
                if (point.get_id() == self.id):
                    return False
                else:
                    return True
            else:
                return False
        else:
            return False


    def find_neighbors(self,points):
        neighbors = []
        for point in points:
            isNeigh = self.isNeighbor(point)  
            if isNeigh:
                neighbors.append(point)
        return neighbors
    
    def check_isDone(self,points):
        isDone = True
        for point in points:
            isVisited = point.get_isVisited()
            if not isVisited:
                return False
        return isDone

    def traverse_tree(self,pointsList):
        pointsInvolved = []
        path = []
        neighbors = self.find_neighbors(pointsList)
        self.isVisited = True

        isDone = self.check_isDone(neighbors)
        if isDone:
            return path,pointsInvolved

        for point in neighbors:
                isVisited = point.get_isVisited()
                if isVisited:
                    continue
                isEnabled = point.get_isEnabled()
                if isEnabled == 1:
                    path.append(point.get_id())
                    pointsInvolved.append(point)
                    path_prev , pointsInvolved_prev = point.traverse_tree(pointsList)
                    path += path_prev
                    pointsInvolved += pointsInvolved_prev
                    return path,pointsInvolved
                else:
                    continue
        if (len(path) == 0):
            for point in neighbors:
                isVisited = point.get_isVisited()
                if isVisited:
                    continue
                isEnabled = point.get_isEnabled()
                if isEnabled == 2:
                    path.append(point.get_id())
                    pointsInvolved.append(point)
                    path_prev , pointsInvolved_prev = point.traverse_tree(pointsList)
                    path += path_prev
                    pointsInvolved += pointsInvolved_prev
                    return path,pointsInvolved
                else:
                    continue
        return path,pointsInvolved

    # Getters    
    def get_isVisited(self):
        return self.isVisited
    
    def get_isEnabled(self):
        return self.isEnabled
    
    def get_id(self):
        return self.id
    
    def get_gridID(self):
        return self.gridID_1,self.gridID_2
    
    def get_pixels(self):
        return self.row,self.col

def crop_image(pts):
    folder_path = "C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Tool\Data"
    all_files = os.listdir(folder_path)
    tif_files = [file for file in all_files if file.lower().endswith('.tif')]
    counter = 0
    for tif_file in tif_files:
        image_path = os.path.join(folder_path, tif_file)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        rows = []
        cols = []
        for pt in pts:
            pixels = pt.get_pixels()
            rows.append(pixels[0])
            cols.append(pixels[1])
        
        left = int(np.min(rows))
        right = int(np.max(rows))
        top = int(np.min(cols))
        bottom = int(np.max(cols))
        width = right-left
        height = bottom - top
        centre_w = int(left + width/2)
        centre_h = int(top + height/2)
        if width < height:
            image_1 = image[left:right,(centre_h-width):(centre_h+width)] 
        else:
            image_1 = image[centre_w-height:centre_w+height,top:bottom]
        resized_image_1 = cv2.resize(image_1[:,:,0], (32,32))
        counter += 1

def crop_resize_image_test(image):
    default_m = 5
    w,h=image.shape[:2]
    image_gray = image 
    #TODO: fix in inputs:
    for im_i in range(w):
        for im_j in range(h):
            if(image_gray[im_i,im_j] <= 10):
                image_gray[im_i,im_j] = 0
    

  
    img_short  = image_gray
    w,h=img_short.shape[:2]
    rows = []
    cols = []
    for m in range(w):
        for n in range(h):
            if(img_short[m,n] != 0):
                rows.append(m)
                cols.append(n)
    else:
        w_L=min(rows)
        w_R=max(rows)
        h_B=min(cols)
        h_T=max(cols)
        centre_x =  (w_L + w_R)/2
        centre_z = (h_B + h_T)/2
        w_lim = int(w_R-w_L)
        h_lim = int(h_T-h_B)
        if (w_lim > h_lim):
            lim = w_lim
        else:
            lim = h_lim
        l = int(np.floor(centre_x - w_lim))
        r = int(np.ceil(centre_x + w_lim))
        b = int(np.floor(centre_z - w_lim))
        t = int(np.ceil(centre_z + w_lim))
        # sprint(l,r,b,t)
        # print('w:',w,' h',h)
        pad_list=[]
        pads = {}
        pads['l'] = l
        pads['r'] = r
        pads['b'] = b
        pads['t'] = t
        data_type = img_short.dtype
        if l < 0:
            pad_list.append('pad_l')
            pads['l'] = 0
        if r > w:
            pad_list.append('pad_r')
            pads['r'] = w
        if b < 0:
            pad_list.append('pad_b')
            pads['b'] = 0
        if t > h:
            pad_list.append('pad_t')
            pads['t'] = h

        cropped_image = img_short[pads['l']:pads['r'],pads['b']:pads['t']]
        if 'pad_l' in pad_list:
            pad_l = np.zeros((abs(l),(pads['t']-pads['b']))).astype(data_type)
            pads['l'] = l
            cropped_image = np.concatenate((pad_l,cropped_image),axis=0)
        if 'pad_r' in pad_list:
            pad_r = np.zeros(((r-w),(pads['t']-pads['b']))).astype(data_type)
            pads['r'] = r
            cropped_image = np.concatenate((cropped_image,pad_r),axis=0)
        if 'pad_b' in pad_list:
            pad_b = np.zeros(((pads['r']-pads['l']),abs(b))).astype(data_type)
            pads['b'] = b
            cropped_image = np.concatenate((pad_b,cropped_image),axis=1)
        if 'pad_t' in pad_list:
            pad_t = np.zeros(((pads['r']-pads['l']),(t-h))).astype(data_type)
            pads['t'] = t
            cropped_image = np.concatenate((cropped_image,pad_t),axis=1)
        
        resized_image = cv2.resize(cropped_image, (32,32)) # (32,32,3)
    # --------------- 
    return resized_image

def apply_identifier(isCropped):
    folder_path = 'C:\\PROJECTS\\BehavioralLandmarks\\BehavioralLandmarks_Python\\Tool\\Data\\NoLandmarks'
    all_files = os.listdir(folder_path)
    tif_files = [file for file in all_files if file.lower().endswith('.tif')]
    loaded_images = []
    my_dict={}
    for tif_file in tqdm(tif_files):
        image_path = os.path.join(folder_path, tif_file)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if isCropped:
            if(len(image.shape) == 3):
                image = image[:,:,0]
            if(image.shape[0] > 32 or image.shape[1] > 32):
                image = cv2.resize(image, (32,32))
        else:
            image = crop_resize_image_test(image)
        loaded_images.append(image)
        my_dict[tif_file] = "None"

    x = np.array(loaded_images)
    outputs = identifier(x)
    prediction = np.argmax(outputs,axis=1)
    behavior_dict={}
    behavior_dict['0']='Approach'
    behavior_dict['1']='Wander'
    behavior_dict['2']='CircleAround'
    behavior_dict['3']='Avoid'
    behavior_dict['4']='BI'
    k=0
    for key, value in my_dict.items():
        my_dict[key]=behavior_dict[str(prediction[k])]
        k += 1

    with open('identifier.json', 'w') as json_file:
        json.dump(my_dict, json_file)

    return my_dict

# Bring it together:
if __name__ == "__main__":
    # Load disctionary:
    with open('C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Models\Detection\detectionDemo.json', 'r') as file:
        det_dict = json.load(file)

    dim_x = 10 # TODO
    dim_y = 10
    n_cols = int(dim_x / 5)
    n_rows = int(dim_y / 5)

    output_dict={}
    for i in range(n_cols):
        for j in range(n_rows):
            output_dict[f'{i}{j}'] = 'BI'


    classifier_T = load_model("C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Models\Classification_Temporal\classifier_Tv2.h5")
    print("Clasifier_T IS LOADED!!")
    identifier = load_model("C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Models\Identification\identifierNew_v2.h5")
    print("Identifier IS LOADED!!")

    output_dict = tackle_rest(output_dict,det_dict)

    # Run "classifier_T" for TemporalLandmarks.
    output_dict = apply_temporal_classifier(output_dict)


    # Create points:
    points , env  = create_env()

    # Find particion
    root = points[0]
    path,pointsInvolved = root.traverse_tree(points)
    path = [root.get_id()] + path
    print("Found Path is: ",path)
    pointsInvolved = [root] + pointsInvolved

    # Run identifier
    apply_identifier(isCropped=False)
    
