# IMPORTS
import numpy as np
from PIL import Image
import os
import cv2
import torch
from tqdm import tqdm
import pickle
import torch.utils.data as data
import tifffile

# HELPER FUNCTIONS
def zoom_image(image):
    min_zoom = 1.1
    max_zoom = 2
    random_zoom = np.random.uniform(min_zoom, max_zoom)
    center_x, center_y = width // 2, height // 2
    scaled_width, scaled_height = int(width * random_zoom), int(height * random_zoom)
    zoom_matrix = np.array([[random_zoom, 0, (1 - random_zoom) * center_x],
                            [0, random_zoom, (1 - random_zoom) * center_y]], dtype=np.float32)
    image_zoomed = cv2.warpAffine(image, zoom_matrix, (width, height), flags=cv2.INTER_LINEAR)
    return image_zoomed
def normalize(image):
    mean = np.mean(image)
    std = np.std(image)
    scaled_image = (image - mean) / std
    return scaled_image
def add_gaussian_noise(image, mean=0, std=0.015):
    row, col = image.shape
    gauss = np.random.normal(mean, std, (row, col))
    noisy = np.clip(image + gauss, 0, 255)
    return noisy.astype(np.float32)

substring = "_a1"
counter = 0

# TEST DATA:
folder_path_3 = 'C:/PROJECTS/BehavioralLandmarks/BehavioralLandmarks_Python/Data/Images/TemporalLandmarks/TestData' #TODO
all_files = os.listdir(folder_path_3)
tif_files = [file for file in all_files if file.lower().endswith('.tif')]
for tif_file in tqdm(tif_files):
    old_name = tif_file.split(substring, 1)[0]
    try:
        counter += 1
        image_path = os.path.join(folder_path_3, tif_file)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) #TYPE 1: TEMPORAL
        np.savez(f'C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Data\PythonFiles\TestDataTemporal\{old_name}_{counter}_type_1_rot_0.npz', image)
    except Exception as e:
        print(f"Error loading image '{tif_file}': {e}")
print("TEMPORAL TEST IMAGES LOADED!")
exit()

# DATA PRE-PROCESSING:
folder_path_3 = 'C:/PROJECTS/BehavioralLandmarks/BehavioralLandmarks_Python/Data/Images/TemporalLandmarks' 
all_files = os.listdir(folder_path_3)
tif_files = [file for file in all_files if file.lower().endswith('.tif')]
for tif_file in tqdm(tif_files):
    old_name = tif_file.split(substring, 1)[0]
    try:
        counter += 1
        image_path = os.path.join(folder_path_3, tif_file)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        np.savez(f'C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Data\PythonFiles\TrainDataTemporal\{old_name}_{counter}_type_1_rot_0.npz', image)
        np.savez(f'C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Data\PythonFiles\TrainDataTemporal\{old_name}_{counter}_noisy_type_1_rot_0.npz', add_gaussian_noise(image))
        np.savez(f'C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Data\PythonFiles\TrainDataTemporal\{old_name}_{counter}_norm_type_1_rot_0.npz', normalize(image))
        np.savez(f'C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Data\PythonFiles\TrainDataTemporal\{old_name}_{counter}_norm_noisy_type_1_rot_0.npz', add_gaussian_noise(normalize(image)))
        height, width = image.shape[:2]
        min_angle = -179
        max_angle = 179
        random_angles = np.random.uniform(min_angle, max_angle, size=10)
        for i, angle in enumerate(random_angles):
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image_rotated = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
            np.savez(f'C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Data\PythonFiles\TrainDataTemporal\{old_name}_{counter}_type_1_rot_{int(angle)}.npz', image_rotated)
            np.savez(f'C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Data\PythonFiles\TrainDataTemporal\{old_name}_{counter}_norm_type_1_rot_{int(angle)}.npz', normalize(image_rotated))
            np.savez(f'C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Data\PythonFiles\TrainDataTemporal\{old_name}_{counter}_noisy_type_1_rot_0.npz', add_gaussian_noise(image_rotated))
            np.savez(f'C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Data\PythonFiles\TrainDataTemporal\{old_name}_{counter}_norm_noisy_type_1_rot_{int(angle)}.npz', add_gaussian_noise(normalize(image_rotated)))
    except Exception as e:
        print(f"Error loading image '{tif_file}': {e}")
print("TEMPORAL IMAGES LOADED!")

exit()