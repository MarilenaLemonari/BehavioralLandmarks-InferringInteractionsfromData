# Imports
import os
import math
import numpy as np
import pandas as pd
import tifffile
import json
from tqdm.auto import tqdm
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2

 #TODO: remove source csv.

# Parameters
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# name = "SpatialLandmarks"
# name = "TemporalLandmarks"
name = "NoLandmarksExt\TestData" #TODO change according to mode
category = "Testing"
# name = "TemporalLandmarks/TestData"
path = "\Trajectories\\" + name + "\\"
csv_directory = current_file_dir + path
row_step = 1
framerate = 10

# Helper functions
def crop_resize_image(image):
    w,h=image.shape[:2]
    rows = []
    cols = []
    for i in range(w):
        for j in range(h):
            if(image[i,j] != 0):
                rows.append(i)
                cols.append(j)
    if(len(rows) == 0 or len(cols) == 0):
        # Disgard black image:
        return 'BI'

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
    pad_list=[]
    pads = {}
    pads['l'] = l
    pads['r'] = r
    pads['b'] = b
    pads['t'] = t
    data_type = image.dtype
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

    cropped_image = image[pads['l']:pads['r'],pads['b']:pads['t']]
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
    
    resized_image = cv2.resize(cropped_image, (32,32))

    return resized_image

def skip_rows(index, step):
    return index % step != 0
    
def iqr_bounds(df, column_name, low, high):
    Q1 = df[column_name].quantile(low)
    Q3 = df[column_name].quantile(high)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

def dist(a, b):
    """Calculates the Euclidean distance between two agents based on their spawn and goal positions."""
    dx = a[1] - b[1]
    dz = a[2] - b[2]
    gx = a[4] - b[4]
    gz = a[5] - b[5]
    return np.sqrt(dx*dx + dz*dz + gx*gx + gz*gz)

def frame_diff(a, b):
    """Calculates the difference in spawn and goal frames between two agents."""
    sf_diff = abs(a[0] - b[0])
    gf_diff = abs(a[3] - b[3])
    return sf_diff + gf_diff

def is_close(a, b, dist_threshold, frame_threshold):
    """Determines whether two agents are 'close' based on their distances and frame differences."""
    return dist(a, b) < dist_threshold and frame_diff(a, b) < frame_threshold

def detect_groups(agent_dict, dist_threshold=0.1, frame_threshold=25):
    group_id = 0
    agent_groups = {}
    for frame in agent_dict:
        for agent in agent_dict[frame]:
            if not agent_groups:
                agent.append(group_id)
                agent_groups[group_id] = [agent]
                group_id += 1
            else:
                added_to_group = False
                for gid, agents in agent_groups.items():
                    if any(is_close(agent, a, dist_threshold, frame_threshold) for a in agents):
                        agent.append(gid)
                        agent_groups[gid].append(agent)
                        added_to_group = True
                        break
                if not added_to_group:
                    agent.append(group_id)
                    agent_groups[group_id] = [agent]
                    group_id += 1
    return agent_dict

def create_frame_dict(data):
    frame_dict = {}
    for human_id, df in data.items():
        for index, row in df.iterrows():
            frame = row['frame']
            pos_x = row['pos_x']
            pos_z = row['pos_z']
            speed = row['speed']
            if frame not in frame_dict:
                frame_dict[frame] = []
            frame_dict[frame].append((pos_z, pos_x, speed))
    return frame_dict

def normalize_array(arr, new_min, new_max):
    normalized_arr = np.zeros_like(arr)
    for channel in range(arr.shape[2]):
        arr_min = np.min(arr[:, :, channel])
        arr_max = np.max(arr[:, :, channel])
        if (arr_max - arr_min) > 0:
            normalized_arr[:, :, channel] = new_min + (arr[:, :, channel] - arr_min) * (new_max - new_min) / (arr_max - arr_min)  
    return normalized_arr

def speed_function(speed):
    return speed ** 3

def thicken_lines(arr, z, x, radius, value):  
    rows, cols = arr.shape
    r_start_row, r_end_row = max(0, z - radius), min(rows, z + radius + 1)
    r_start_col, r_end_col = max(0, x - radius), min(cols, x + radius + 1)

    for row in range(r_start_row, r_end_row):
        for col in range(r_start_col, r_end_col):
            distance = max(abs(row - z), abs(col - x))
            arr[row, col] += value / (2 ** distance)

def adjust_color(name):
    my_array = io.imread(f'{name}.tif')
    resolution_x=my_array.shape[0]
    resolution_y=my_array.shape[1]
    trajectories=1-np.expand_dims(my_array[:,:,0],axis=-1)
    environment=np.expand_dims(my_array[:,:,1],axis=-1)
    pad=np.zeros((resolution_x,resolution_y,1))
    image=np.ones((resolution_x,resolution_y,3))
    for i in range(resolution_x):
        for j in range(resolution_y):
            if environment[i,j,0]!=0:
                #Not Background
                if trajectories[i,j,0]==1:
                    #Background
                    image[i,j,:]=np.array([0.65,0,0])
            else:
                #Background
                if trajectories[i,j,0]!=1:
                    #Not background
                    image[i,j,:]=np.array([ trajectories[i,j,0],trajectories[i,j,0],trajectories[i,j,0] ])
    plt.imshow(image)
    plt.tick_params(length=0, labelbottom=False, labelleft=False)
    plt.savefig(f'{name}.png')

def process_frame_range(args):
    key, frame_dict, env_dict, step, separation, dataset_name, kernel_size, stride, thicken_radius, start_index, end_index = args
    
    dir_name = current_file_dir + "\Images\\" + dataset_name + "\\"
    os.makedirs(dir_name, exist_ok=True)
    dir_full_name = dir_name

    empty_images_predictions = {}
    width, height = 256, 256

    values = np.zeros((height, width, 1), np.float32)
    agents_per_pixel = np.zeros((height, width, 1), np.float32)

    frame_keys = list(frame_dict.keys())[start_index:end_index]
    for frame_key in frame_keys:
        for data in frame_dict[frame_key]:
            x = math.floor(data[1] * width) - 1
            z = math.floor(data[0] * height) - 1
            speed = data[2]
            if speed > 0:
                values[z][x] += np.clip(1 - (speed_function(speed) / 1.0), 0, 1)
                agents_per_pixel[z][x] += 1

    img = np.zeros((height, width, 2), np.float32)
    for z in range(height):
        for x in range(width):
            if agents_per_pixel[z][x] > 0:
                val = values[z][x] / agents_per_pixel[z][x]
                thicken_lines(img[:,:,0], z, x, radius=thicken_radius, value=val)

    # Draw objects
    for obj in env_dict:
        scale_x = obj['scale_x']
        scale_z = obj['scale_z']
        x = math.floor(obj['pos_x'] * width) - 1
        z = math.floor((1 - obj['pos_z']) * height) - 1
        range_x = math.floor(scale_x * width) // 2
        range_z = math.floor(scale_z * height) // 2
        env_object = 1 if obj['type'] > 0.5 else 0
        z_min_range, z_max_range = np.clip(z - range_z, 0, height), np.clip(z + range_z + 1, 0, height)
        x_min_range, x_max_range = np.clip(x - range_x, 0, width), np.clip(x + range_x + 1, 0, width)
        img[z_min_range:z_max_range, x_min_range:x_max_range, 1] = env_object
        img[z_min_range:z_max_range, x_min_range:x_max_range, 0] = 0

    img = normalize_array(img, 0, 1)
    img = np.flip(img, axis=0)

    img_out=img[:,:,0]
    
    filename = 'img_'+ key
    
    resized_image = crop_resize_image(img_out)
    if isinstance(resized_image, str) == False:
        tifffile.imwrite(dir_full_name + filename + '.tif', resized_image)

    pad = np.zeros((256, 256, 1), np.float32)
    for j in range(separation[0]):
        for i in range(separation[1]):
            parent_size = 256 * kernel_size
            img_parent = img[i*parent_size:(i+1)*parent_size, j*parent_size:(j+1)*parent_size, :]

            for jj in range(256 // stride):
                for ii in range(256 // stride):
                    filename = 'img_' + str(start_index) + '_' + str(end_index) + '_' + str(i) + '_' + str(j) + '_' + str(ii) + '_' + str(jj)
                    img_out = img_parent[ii*stride:256+ii*stride, jj*stride:256+jj*stride, :]                
                    # If image does not have enought data, do not generate and set predction to -1
                    if np.sum(img_out[:,:,0] > 0) / (256 * 256) < 0.05:
                        empty_images_predictions[filename + ".npz"] = -1
                    else:
                        # Save npz file       
                        np.savez_compressed(dir_name + filename, img_out)  
                        # add 3rd channel and save image
                        img_out = np.concatenate((img_out, pad), axis=2)

    return empty_images_predictions

def parallel_create_images(key,frame_dict, env_dict, step, separation, dataset_name, kernel_size, stride, thicken_radius, n_processes=None):
    if n_processes is None:
        n_processes = os.cpu_count()

    counts = math.ceil(len(frame_dict) / step)
    tasks = []

    for count in range(counts):
        start_index, end_index = step * count, step * (count + 1)
        tasks.append((key,frame_dict, env_dict, step, separation, dataset_name, kernel_size, stride, thicken_radius, start_index, end_index))

    all_empty_predictions = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
        results = list(tqdm(executor.map(process_frame_range, tasks), total=len(tasks), desc="Images Generation"))

    for empty_predictions in results:
        all_empty_predictions.update(empty_predictions)

    return all_empty_predictions

def read_csv_files_and_env_json(csv_directory,row_step,framerate):
    with open(csv_directory + "env.json") as json_file:
        data_json = json.load(json_file)
        scene_objects_dict = data_json["EnvironmentObjects"]
        env_size_dict = data_json['EnvironmentParams']
    
    pos_x_min, pos_x_max = env_size_dict['min_width'], env_size_dict['max_width']
    pos_z_min, pos_z_max = env_size_dict['min_height'], env_size_dict['max_height']

    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

    # Create an empty dictionary to store the data and a list to store all DataFrames for concatenation
    data_dict = {}
    all_dfs = []

    # Files with number of rows less thatn row_threshold will not be included
    row_threshold = 3

     # Iterate through the CSV files and read the content into the dictionary
    for filename in csv_files:
        # Read the CSV file into a pandas DataFrame and assign column names
        df = pd.read_csv(os.path.join(csv_directory, filename), 
            header=None, names=['frame', 'pos_x', 'pos_z','or_x','or_z'], 
            skiprows=None,
            #skiprows=lambda index: index == 0 or skip_rows(index, row_step),
            usecols=[0, 1, 2])
        if df.shape[0] < row_threshold:
            continue
        # Calculate the speed based on consecutive positions
        df['distance'] = np.sqrt((df['pos_x'].diff())**2 + (df['pos_z'].diff())**2)
        df['time_diff'] = df['frame'].diff()
        df['speed'] = df['distance'] / df['time_diff']
        df.dropna(inplace=True)  # Remove the first row with NaN values due to the diff() calculations
        
        # Drop the 'distance' and 'time_diff' columns, as they are not needed anymore
        df.drop(columns=['distance', 'time_diff'], inplace=True)
        # Store the DataFrame in the dictionary using the filename as the key
        data_dict[filename] = df
        all_dfs.append(df)

    # Concatenate all DataFrames into one large DataFrame
    all_data = pd.concat(all_dfs, ignore_index=True)

    # Calculate the lower and upper bounds for pos_x, pos_z, and speed columns
    pos_x_bounds = iqr_bounds(all_data, 'pos_x', 0.2, 0.8)
    pos_z_bounds = iqr_bounds(all_data, 'pos_z', 0.2, 0.8)
    speed_bounds = (all_data['speed'].quantile(0.1), all_data['speed'].quantile(0.9))
    speed_min, speed_max = speed_bounds[0], speed_bounds[1]

    bound=15
    pos_x_min=-bound
    pos_x_max=bound
    pos_z_min=-bound
    pos_z_max=bound

    agent_dict = {}
    frame_interval = 1 / framerate
    for filename, df in data_dict.items():
        # Normalize columns
        df = df[(df['pos_x'] >= -bound) & (df['pos_x'] <= bound)]
        df = df[(df['pos_z'] >= -bound) & (df['pos_z'] <= bound)]
        # df = df[(df['pos_x'] >= pos_x_bounds[0]) & (df['pos_x'] <= pos_x_bounds[1])]
        # df = df[(df['pos_z'] >= pos_z_bounds[0]) & (df['pos_z'] <= pos_z_bounds[1])]
        # # Normalize columns
        df['pos_x'] = (df['pos_x'] - pos_x_min) / (pos_x_max - pos_x_min)
        df['pos_z'] = (df['pos_z'] - pos_z_min) / (pos_z_max - pos_z_min)
        # df['speed'] = ((df['speed'] - speed_min) / (speed_max - speed_min)).clip(0, 1)

        start_frame = (int)(df['frame'].iloc[0] / frame_interval)
        start_pos_x, start_pos_z = df['pos_x'].iloc[0], df['pos_z'].iloc[0]
        end_frame = (int)(df['frame'].iloc[-1] / frame_interval)
        end_pos_x, end_pos_z = df['pos_x'].iloc[-1], df['pos_z'].iloc[-1] 
        agent_data = [start_frame, start_pos_x, start_pos_z, end_frame, end_pos_x, end_pos_z]
        if start_frame not in agent_dict:
                agent_dict[start_frame] = []
        agent_dict[start_frame].append(agent_data)

        data_dict[filename] = df

    sorted_agent_dict = {k: agent_dict[k] for k in sorted(agent_dict, key=int)}
    
    final_agent_dict = detect_groups(sorted_agent_dict)

    return data_dict, scene_objects_dict, final_agent_dict

def get_grid_separation(width, height, multiplier):
    # Calculate the greatest common divisor (GCD) of the width and height
    divisor = math.gcd(width, height)
    # Divide the width and height by the GCD
    width_normalized = width / divisor
    height_normalized = height / divisor
    # Find the larger value between width and height
    larger_value = max(width_normalized, height_normalized)
    # Divide both width and height by the larger value
    width_normalized /= larger_value
    height_normalized /= larger_value

    width_normalized = math.floor(multiplier * width_normalized)
    height_normalized = math.floor(multiplier * height_normalized)

    return (width_normalized, height_normalized)

# Execute
if __name__ ==  '__main__':
    csv_data, env_dict, agent_dict = read_csv_files_and_env_json(csv_directory,row_step,framerate)

    frame_interval = 200
    kernel_size = 2
    stride = 1
    video_width = 600
    video_height = 480
    grid_multiplier = 4
    separation = get_grid_separation(video_width, video_height, grid_multiplier)
    thicken_radius = 2
    n_processes = 16

    n_csvs = len(csv_data)
    dict_list = list(csv_data.items())

    for i in range(n_csvs):
        key, value = dict_list[i]
        csv_data_ind = {key: value}
        frame_dict = create_frame_dict(csv_data_ind)
        prefix = 'img_'+key.split("_")[0]
        folder_path = "C:\\PROJECTS\\BehavioralLandmarks\\BehavioralLandmarks_Python\\Data\\Images\\NoLandmarksExt\\TestData"
        files = os.listdir(folder_path)
        file_exists = any(file.startswith(prefix) for file in files)
        if file_exists == False:
            # Change structure of dictionary to add points of agents in each frame
            empty_predictions = parallel_create_images(key, frame_dict, env_dict, frame_interval, separation, name, kernel_size, stride, thicken_radius, n_processes=n_processes)