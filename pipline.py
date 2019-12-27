# coding: utf-8

# Step 1: first pip install packages

# !pip install *.whl

# Step 2: import packages

import os
import glob
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import gc
import sys

from tqdm.notebook import tqdm
import subprocess as sp

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
import keras

import tensorflow as tf

# Step 3: Define GPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set GPU memory 
if ('tensorflow' == K.backend()):
    from keras.backend.tensorflow_backend import set_session

#     gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
#     config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

# plot inline
%matplotlib inline

# Step 4: Define Model Class

class Model:
    def __init__(self):
        pass

# Step 5: Load Model

model = Model()
state_dict = torch.load(pth_path))
# If load failed, maybe to update state_dict
for name, weights in state_dict.items():
    state_dict[name] = weights

model.load_state_dict(state_dict)

# update the model output (if need)
num_fits = model.fc.in_features
model.fc = nn.Linear(num_fits, your_num)

# set eval and to gpu
model.eval().to(device)

# Step 6: Process Data

filenames = glob.glob('/PATH/*.data')
f_lst = tqdm(filenames)  # Show progress bar

result_lst = []

with torch.no_grad():
    # If you need, you can add try...except..., :)
    for filename in f_lst:
        # Preprocessing Data 
        data = preprocess(filename)
        # Predict
        output = model.predict(data)
        # Post-processing
        result = post_processing(output)
        # save result
        result_lst.append(result)

# Visualization (Optional)

def visual(input_data):
    im = Image.fromarray(input_data.astype(np.uint8))
    plt.imshow(im, cmap='gray')
    plt.show()


for result in result_lst:
    # processing data
    input_data = process(result)
    visual(input_data)

# Step 7: Save output
submission = []
for filename, result in zip(filenames, result_lst):
    if result:
        prob = result
    else:
        prob = 0.5
    submission.append([os.path.basename(filename), prob])

submission = pd.DataFrame(submission, columns=['filename', 'label'])


# Dataframe Processing (Optional)
for i in range(len(submission)):
    fn = submission.filename.values[i]
    val = submission.label.values[i]
    # To-do something
    

submission.sort_values('filename').to_csv('submission.csv', index=False)