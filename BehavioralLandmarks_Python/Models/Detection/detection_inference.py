# IMPORTS
from importlib.metadata import requires
import os
from turtle import right
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
import csv

# PERFECT DATA:
# cd C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Data
# python3 C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Models\Detection\detection_inference.py
# IN-THE-WILD:
# go to cd C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Models\Detection

def scale_to_standard_normal(images):
    mean = np.mean(images)
    std = np.std(images)
    scaled_images = (images - mean) / std
    return scaled_images
def crop_resize_image_test(image,user_h,user_v,image_name,imgs):
    default_m = 5
    w,h=image.shape[:2]
    image_gray = image[:,:,0]
    #TODO: fix in inputs:
    for im_i in range(w):
        for im_j in range(h):
            if(image_gray[im_i,im_j] <= 10):
                image_gray[im_i,im_j] = 0
    
    # HERE DISCRETIZE:
    ratio_h = user_h / default_m
    ratio_v = user_v / default_m
    
    pixels_h = int(h / ratio_h)
    pixels_v = int(w / ratio_v)
    
    # imgs = []
    for i in range(int(ratio_h)):
        for j in range(int(ratio_v)):
            img_short  = image_gray[j*(pixels_v): (j+1)*(pixels_v), i*(pixels_h): (i+1)*(pixels_h)]
            w,h=img_short.shape[:2]
            rows = []
            cols = []
            for m in range(w):
                for n in range(h):
                    if(img_short[m,n] != 0):
                        rows.append(m)
                        cols.append(n)
            if(len(rows) == 0 or len(cols) == 0):
                #print('Blank Image')
                my_dict[f'{image_name}_{i}{j}'] = 'BI'
            else:
                if img_short.shape[0] <= 32 and img_short.shape[1] <= 32:
                    print("Check Image Dimensions: Too small")
                    exit()
                
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
                imgs.append(resized_image)
                # import matplotlib.pyplot as plt
                # plt.imshow(cropped_image, cmap='gray')
                # plt.show()
                my_dict[f'{image_name}_{i}{j}'] =  resized_image
                # tifffile.imwrite(f'{image_name}_{i}{j}.tif', resized_image)
    # --------------- 
    return my_dict, imgs


detector = load_model("C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Models\Detection\detectorNew_v1.h5")
print("MODEL IS LOADED!!")

# INFERENCE ON REAL DATA
# inputs = np.random.random((1, 32, 32, 1))
# outputs =  identifier(inputs)
# print(outputs)
# exit()

# # PERFECT DATA:
# folder_path = 'PythonFiles\TestData'  
# file_list = os.listdir(folder_path)
# npz_files = [file for file in file_list if file.endswith('.npz')]
# loaded_images = []
# landmark_type = []
# for npz_file in tqdm(npz_files):
#     name_parts = npz_file.split('_')
#     type_index = name_parts.index("type")
#     value_after_type = name_parts[type_index + 1]
#     file_path = os.path.join(folder_path, npz_file)
#     loaded_data = np.load(file_path)
#     array_keys = loaded_data.files
#     array_key = array_keys[0]
#     array = loaded_data[array_key]
#     if (array.dtype != 'float32'):
#         print(file_path)
#         exit()
#     loaded_images.append(array)
#     landmark_type.append(int(value_after_type))

# gt = np.array(landmark_type)
# x = scale_to_standard_normal(loaded_images)
# outputs = detector(x)
# prediction = np.argmax(outputs,axis=1)
# print(prediction.shape)
# confusion = confusion_matrix(gt, prediction)
# print("Confusion Matrix for Test Data:")
# print(confusion)
# exit()

# REAL DATA
# image = cv2.imread("test.png", cv2.IMREAD_UNCHANGED)
# print(image.shape)
# x = np.array(image)
# out = detector(x)
# print(out)
# exit()

#Load Image:
folder_path = os.getcwd() 
all_files = os.listdir(folder_path)
tif_files = [file for file in all_files if file.lower().endswith('.tif')]
loaded_images = []
index = 0
my_dict={}
# csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
# for tif_file in tqdm(tif_files):
for tif_file in tqdm(tif_files):
    csv_prefix = tif_file.split(".")[0]
    dims = csv_prefix.split("_")[1]
    dim_x = dims.split("x")[0]
    dim_y = dims.split("x")[1]
    # for csv_cand in csv_files:
    #     if csv_prefix in csv_cand:
    #         csv_file = csv_cand
    #     else:
    #         print("NO CSV FOUND")
    #         exit()
    # data_list=[]
    # with open(csv_file, 'r') as file:
    #     csv_reader = csv.reader(file)
    #     for row in csv_reader:
    #         data_list.append([float(row[0]), float(row[1]), float(row[3])])
    # traj_array = np.array(data_list)
    image_path = os.path.join(folder_path, tif_file)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # resized_image = cv2.resize(image, (32,32)) # (32,32,3)
    # if tif_file == "Celine11_5x5.tif_00.tif":
    #     image_float = resized_image.astype(np.float64)
    #     resized_image = image_float / 255.0
    # else:
    #     resized_image *= 255
    #     tifffile.imwrite('prospatho.tif', resized_image)
    # print(tif_file, "details: ", np.max(resized_image), np.min(resized_image))

    # loaded_images.append(resized_image)
    # my_dict[f'{tif_file}'] =  resized_image
    
    my_dict, loaded_images = crop_resize_image_test(image,int(dim_x),int(dim_y),tif_file,loaded_images) #TODO: change user_horizontal, user_vertical
    
# x = scale_to_standard_normal(loaded_images)
x = np.array(loaded_images)
print(x.shape)
outputs = detector(x)

prediction = np.argmax(outputs,axis=1)
print(outputs)
my_landmarks = {}
my_landmarks['0'] = "SpatialLandmarks"
my_landmarks['1'] = "TemporalLandmarks"
my_landmarks['2'] = "NoLandmarks"

k=0
for key, value in my_dict.items():
    if isinstance(value, str) == False:
        my_dict[key]=my_landmarks[str(prediction[k])]
        tifffile.imwrite(f'C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Tool\Data\{my_landmarks[str(prediction[k])]}\{key}.tif', value)
        k += 1

with open('detectionDemo.json', 'w') as json_file:
    json.dump(my_dict, json_file)