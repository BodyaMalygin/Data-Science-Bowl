import os
import sys
import numpy as np
from skimage.transform import resize
from skimage.io import imread
from tqdm import tqdm

def get_dataset(train_path):
    train_ids = next(os.walk(train_path))[1]
    X_train = np.zeros((len(train_ids), 256, 256, 3), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), 256, 256, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = train_path + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:3]
        img = resize(img, (256, 256), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((256, 256, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (256, 256), mode='constant', 
                                        preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask
    print('Done!')

    return X_train, Y_train