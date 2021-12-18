# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 16:49:12 2021

@author: Zaman
"""

import os
import sys
from tqdm import tqdm
import random

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize

    
    
def img_preprocess(img_dir_path, target_shape=(128, 128, 3)):
    
    IMG_HEIGHT, IMG_WIDTH, IMG_CH = target_shape
    ids = next(os.walk(img_dir_path))[1]
    
    images = np.zeros((len(ids), IMG_HEIGHT, IMG_WIDTH, IMG_CH), dtype=np.uint8)
    
    sys.stdout.flush()

    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        image_name =  id_ +  '.png'
        img_path = os.path.join(img_dir_path, id_, 'images',image_name)
        single_img = imread(img_path)[:,:,:IMG_CH] # image shape(H,W,4) -> image shape(H,W,3)
        single_img = resize(single_img, (IMG_HEIGHT, IMG_WIDTH, IMG_CH), preserve_range=True, mode = 'constant')
        images[n] = single_img
        
    return images

def mask_preprocess(img_dir_path, target_shape=(128, 128, 1)):
    
    IMG_HEIGHT, IMG_WIDTH, IMG_CH = target_shape
    ids = next(os.walk(img_dir_path))[1]
    y_train = np.zeros((len(ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    sys.stdout.flush()

    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        mask_dir_path = os.path.join(img_dir_path, id_, 'masks')
        for mask_image in (next(os.walk(mask_dir_path))[2]):
            single_mask_path = os.path.join(mask_dir_path, mask_image)
            single_mask = imread(single_mask_path)
            single_mask = resize(single_mask, (IMG_HEIGHT, IMG_WIDTH), preserve_range=True, mode = 'constant')
            single_mask = np.expand_dims(single_mask, axis=-1)
            mask = np.maximum(mask, single_mask)
            
        y_train[n] = mask 
        
    return y_train

    
    
if __name__ =='__main__':
    
    # set seed
    seed = 42
    random.seed = seed
    np.random.seed = seed
    
    
    # Set paths
    TRAIN_IMG_PATH = os.path.join('data', 'stage1_train')
    TEST_IMG_PATH = os.path.join('data', 'stage1_test')
    
    X_train = img_preprocess(TRAIN_IMG_PATH,(128,128,3))
    y_train = mask_preprocess(TRAIN_IMG_PATH, (128,128,1))
    X_test = img_preprocess(TEST_IMG_PATH, (128,128,3))
    
    fig, arr = plt.subplots(1,2, figsize=(15,15))
    ix = random.randint(0, 50)
    arr[0].imshow(X_train[ix])
    arr[1].imshow(y_train[ix][:,:,0])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    