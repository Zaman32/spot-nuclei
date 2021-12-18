# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 01:49:40 2021

@author: Zaman
"""

import tensorflow
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Lambda
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model


def encoder_block(pre_layer, n_filters = 32,  dropout_prob=0.3, max_pooling = True):
    en_layer = Conv2D(n_filters, 
                   kernel_size=(3,3) , 
                   padding='same', 
                   activation='relu',
                   kernel_initializer='HeNormal')(pre_layer)
    
    en_layer = Conv2D(n_filters, 
                   kernel_size=(3,3) , 
                   padding='same',
                   activation='relu',
                   kernel_initializer='HeNormal')(en_layer)
     
    en_layer = BatchNormalization()(en_layer, training=False)
        
    
    if dropout_prob>0:
        en_layer = Dropout(dropout_prob)(en_layer)
    
    skip_connection = en_layer    
    
    if max_pooling:
        en_layer = MaxPool2D(pool_size=(2,2))(en_layer)
    
    return en_layer, skip_connection


def decoder_unit(prev_layer, skip_connection, n_filters = 32):
    
    de_layer = Conv2DTranspose(n_filters, 
                               kernel_size=(2,2), 
                               strides=(2,2),
                               kernel_initializer='HeNormal',
                               padding='same',
                               activation='relu')(prev_layer)
    
    de_layer = concatenate([skip_connection, de_layer], axis=3)
    
    de_layer = Conv2D(n_filters, 
                   kernel_size=(3,3) , 
                   padding='same',
                   activation='relu',
                   kernel_initializer='HeNormal')(de_layer)
    
    de_layer = Conv2D(n_filters, 
                   kernel_size=(3,3) , 
                   padding='same',
                   activation='relu',
                   kernel_initializer='HeNormal')(de_layer)
    
    return de_layer


def unet_model(input_shape, n_filters, n_class):
    ip_layer = Input(input_shape)
    
    ip_layer = Lambda(lambda x: x/255)(ip_layer)
    
    en1 = encoder_block(ip_layer, n_filters, dropout_prob=0.2)
    en2 = encoder_block(en1[0], 2*n_filters, dropout_prob=0.2)
    en3 = encoder_block(en2[0], 4*n_filters, dropout_prob=0.2)
    en4 = encoder_block(en3[0], 8*n_filters, dropout_prob=0.2)
    en5 = encoder_block(en4[0], 16*n_filters, dropout_prob=0.2, max_pooling=(False))
    
    de1 = decoder_unit(en5[0], en4[1], 8*n_filters)
    de2 = decoder_unit(de1, en3[1], 4*n_filters)
    de3 = decoder_unit(de2, en2[1], 2*n_filters)
    de4 = decoder_unit(de3, en1[1], n_filters)
    
    out = Conv2D(n_class, (1,1), activation='sigmoid')(de4)
    
    model = Model(inputs=[ip_layer], outputs=[out])
    
    return model

if __name__ == '__main__':
    
    unet = unet_model((128,128,3), 16, 1)
    
    unet.summary()
    
                                   
                                   
                                 
                                   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    