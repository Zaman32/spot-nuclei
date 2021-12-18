# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 16:27:30 2021

@author: Zaman
"""

from data_processing import img_preprocess, mask_preprocess
from model import encoder_block, decoder_unit, unet_model
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf 
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model 

if __name__ == '__main__':
    

    # set gpu limit
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # set seed
    seed = 42
    random.seed = seed
    #np.random.seed(seed=seed)
        
    
    # Set paths
    TRAIN_IMG_PATH = os.path.join('data', 'stage1_train')
    TEST_IMG_PATH = os.path.join('data', 'stage1_test')
    MODEL_PATH = os.path.join('Model', 'unet_model.h5')
    
    # Data preprocessing
    X_train = img_preprocess(TRAIN_IMG_PATH,(128,128,3))
    y_train = mask_preprocess(TRAIN_IMG_PATH, (128,128,1))
    X_test = img_preprocess(TEST_IMG_PATH, (128,128,3))
    
    # Visualize processed data
    fig, arr = plt.subplots(1,2, figsize=(15,15))
    ix = random.randint(0, 50)
    arr[0].imshow(X_train[ix])
    arr[1].imshow(y_train[ix][:,:,0])
    
    TRAIN = False
    TEST = True
    
    if TRAIN:
        # Create model
        unet = unet_model((128,128,3), 16, 1)
    
        # setting callbacks
        early_stop = EarlyStopping(patience=5, verbose=1)
        
        save_ceckpoint = ModelCheckpoint(MODEL_PATH,
                                         verbose=1,
                                         save_best_only=(True)
                                         )
    
    
        unet.compile(optimizer='adam', 
                     loss='binary_crossentropy', 
                     metrics=['accuracy']
                     )
        
        unet.fit(X_train, y_train, batch_size=20, epochs = 50, 
                 callbacks=[early_stop, save_ceckpoint], 
                 validation_split=0.1
                 )
    
        unet.summary()
    
    if TEST:
        model = load_model(MODEL_PATH)
        test_img = X_test[random.randint(0,len(X_test))]
        pred_y = model.predict(test_img[np.newaxis, :, :, :])
        new_pred = np.squeeze(pred_y, axis=0)
        
        fig, arr = plt.subplots(1,2, figsize=(15,15))
        arr[0].imshow(test_img)
        arr[1].imshow(new_pred)                          
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    