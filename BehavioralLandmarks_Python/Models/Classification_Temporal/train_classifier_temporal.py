# IMPORTS
from importlib.metadata import requires
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.layers import Dropout

os.environ['WANDB_API_KEY']="..."

#TODO: go to cd C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Data
# Execute python3 C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Models\Classification_Temporal\train_classifier_temporal.py

# HELPER FUNCTIONS
def visualize_image(array):
    image_rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title('Image')
    plt.axis('off')
    plt.show()
def scale_to_standard_normal(images):
    mean = np.mean(images)
    std = np.std(images)
    scaled_images = (images - mean) / std
    return scaled_images

# LOAD DATA
folder_path = 'PythonFiles\TrainDataTemporal'  
file_list = os.listdir(folder_path)
npz_files = [file for file in file_list if file.endswith('.npz')]
loaded_images = []
temporal_SWITCH = []
class_1 = 0
class_2 = 0
class_3 = 0
class_4 = 0
class_5 = 0
class_6 = 0
class_7 = 0
class_8 = 0
class_9 = 0
for npz_file in tqdm(npz_files):

    name_parts = npz_file.split('_')
    type_index = name_parts.index("type")
    value_after_type = name_parts[type_index + 1]
    if(int(value_after_type) == 1):# TEMPORAL LANDMARK
        # Read image:
        file_path = os.path.join(folder_path, npz_file)
        loaded_data = np.load(file_path)
        array_keys = loaded_data.files
        array_key = array_keys[0]
        array = loaded_data[array_key]
        if (array.dtype != 'float32'):
            print(file_path)
            exit()
        loaded_images.append(array)

        # Read field id:
        type_index2 = npz_file.find("T")
        value_after_T = int(npz_file[type_index2 + 1])
        temporal_SWITCH.append(value_after_T-1) #TODO: revert for prediction
        if (value_after_T == 1):
            class_1 += 1
        elif (value_after_T == 2):
            class_2 += 1
        elif (value_after_T == 3):
            class_3 += 1
        elif (value_after_T == 4):
            class_4 += 1
        elif (value_after_T == 5):
            class_5 += 1
        elif (value_after_T == 6):
            class_6 += 1
        elif (value_after_T == 7):
            class_7 += 1
        elif (value_after_T == 8):
            class_8 += 1
        elif (value_after_T == 9):
            class_9 += 1
        else:
            print("ERROR: Invalid Temporal Switch.")
            exit()

# print(class_1,class_2,class_3,class_4,class_5,class_6,class_7,class_8,class_9)
# exit()

# NORMALIZE DATA
gt = np.array(temporal_SWITCH)
x = scale_to_standard_normal(loaded_images)

# DATA LOADER
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = torch.from_numpy(image)

        return image, label

dataset = CustomDataset(x,gt)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

x_train, x_val, y_train, y_val = train_test_split(x, gt, test_size=0.2, random_state=42)

# MODEL
model = Sequential()

# Convolutional layers
model.add(Conv2D(16, kernel_size=3, strides=1, padding='same', input_shape=(32, 32, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.15))  # Add dropout (adjust the dropout rate as needed)

model.add(Conv2D(32, kernel_size=3, strides=1, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.15))  # Add dropout (adjust the dropout rate as needed)

# Flatten the 3D feature maps to 1D
model.add(Flatten())

# Fully connected layers with L2 regularization
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.25))  # Add dropout (adjust the dropout rate as needed)

model.add(Dense(128)) 
model.add(Activation('relu'))
model.add(Dropout(0.25))  # Add dropout (adjust the dropout rate as needed)

# Output layer for multi-class classification
model.add(Dense(9))
model.add(Activation('softmax'))

# Creating an instance of the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# HYPERPARAMETERS
wandb.init(project="BehavioralLandmarks")
config = wandb.config
config.epochs = 300
config.batch_size = batch_size

# TRAINING
model.fit(x_train, y_train, epochs=config.epochs, batch_size=config.batch_size,
          validation_data=(x_val, y_val),
          callbacks=[WandbCallback()])
model.save("C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Models\Classification_Temporal\classifier_Tv3.h5")
print("MODEL IS SAVED!!")
wandb.finish()

# CONFUSION MATRICES:

y_train_pred = model.predict(x_train)
y_train_pred_classes = y_train_pred.argmax(axis=-1)
confusion = confusion_matrix(y_train, y_train_pred_classes)
print("Confusion Matrix for Training Data:")
print(confusion)

y_val_pred = model.predict(x_val)
y_val_pred_classes = y_val_pred.argmax(axis=-1)
confusion = confusion_matrix(y_val, y_val_pred_classes)
print("Confusion Matrix for Validation Data:")
print(confusion)