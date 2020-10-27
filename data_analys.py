import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize

TRAIN_PATH = 'data-science-bowl-2018/stage1_train/'
TEST_PATH = 'data-science-bowl-2018/stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 30
random.seed = seed
np.random.seed = seed

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

train_img = []
train_masks = []

train_img_data = []
train_mask_data = []

test_img = []
test_img_data = []

print("Reading train images and masks and getting their metadata")
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread('{}/images/{}.png'.format(path, id_))
    
    train_img.append(img)
    img_height = img.shape[0]
    img_width = img.shape[1]
    
    nucleus_count = 0
    
    for mask_file in next(os.walk('{}/masks/'.format(path)))[2]:
        mask = imread('{}/masks/{}'.format(path, mask_file))
        train_masks.append(mask)
        mask_height, mask_width = mask.shape
        
        train_mask_data.append([n,mask_height, mask_width])
        
        nucleus_count += 1
        
    train_img_data.append([id_, img_height, img_width, nucleus_count])

print("Reading test images and getting metadata")
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(train_ids)):
    path = TEST_PATH + id_
    img = imread('{}/images/{}.png'.format(path, id_))
    
    print(img.shape)
    test_img.append(img)
    img_height = img.shape[0]
    img_width = img.shape[1]
    
    test_img_data.append([id_, img_height, img_width, nucleus_count])