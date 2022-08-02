#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob
import keras
from keras_video import VideoFrameGenerator
import keras_video.utils
import tensorflow as tf
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, BatchNormalization,MaxPool2D, GlobalMaxPool2D
from keras.layers import TimeDistributed, GRU, Dense, Dropout, Masking, Embedding, LSTM ,Flatten
from keras.layers import Input, Conv2D, DepthwiseConv2D,       Dense, Concatenate, Add, ReLU, BatchNormalization, AvgPool2D, LayerNormalization,     MaxPool2D, GlobalAvgPool2D, Reshape, Permute, Lambda, Activation,RepeatVector
from keras import layers
from keras import models
import keras.backend as K
from keras.models import Model
import random
import math
import numpy as np
import pandas as pd
from keras.layers.merge import concatenate


classes = [i.split(os.path.sep)[6] for i in glob.glob(r'/content/drive/My Drive/Colab Notebooks/xxx/*')]
classes.sort()
# some global params
SIZE = (60, 60)
CHANNELS = 3
NBFRAME = 5
BS = 48
#ern to get videos and classes
glob_pattern="/content/drive/My Drive/Colab Notebooks/xxx/{classname}/*.mp4"

# for data augmentation
data_aug = keras.preprocessing.image.ImageDataGenerator(
    zoom_range=.1,
    horizontal_flip=True,
    rotation_range=8,
    width_shift_range=.2,
    height_shift_range=.2)
# Create video frame generator
train = keras_video.generator.VideoFrameGenerator(
    classes=classes, 
    glob_pattern=glob_pattern,
    nb_frames=NBFRAME,
    split_val=.33, 
    shuffle=False,
    batch_size=BS,
    target_shape=SIZE,
    nb_channel=CHANNELS,
    transformation=data_aug,
    use_frame_cache=True)
valid = train.get_validation_generator()


def mobile_net(shape=(60, 60, 3)):
    def mobilenet_block(x, f, s=1):
        x = DepthwiseConv2D(3, strides=s, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    
        x = Conv2D(f, 1, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
    
    
    input = Input(shape)

    x = Conv2D(30, 3, strides=2, padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = mobilenet_block(x, 64)
    x = BatchNormalization()(x)
    x = mobilenet_block(x, 128, 2)
    x = BatchNormalization()(x)
    x = mobilenet_block(x, 128)
    x = BatchNormalization()(x)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = mobilenet_block(x, 256, 2)
    x = BatchNormalization()(x)
    x = mobilenet_block(x, 256)
    x = BatchNormalization()(x)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = mobilenet_block(x, 512, 2)
    for _ in range(5):
        x = mobilenet_block(x, 512)

    x = mobilenet_block(x, 1024, 2)
    x = mobilenet_block(x, 1024)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    output1 = GlobalMaxPool2D()(x)
    model = Model(input,output1)
    return model


def Squeeze_net(shape=(60, 60, 3)):
    def fire(x, fs, fe):
        s = Conv2D(fs, 1, activation='relu')(x)
        e1 = Conv2D(fe, 1, activation='relu')(s)
        e3 = Conv2D(fe, 3, padding='same', activation='relu')(s)
        output = Concatenate()([e1, e3])
        return output
  
  
    input = Input(shape)
  
    x = Conv2D(30, 7, strides=2, padding='same', activation='relu')(input)
    x = BatchNormalization()(x)
    x = MaxPool2D(3, strides=2, padding='same')(x)
  
    x = fire(x, 16, 64)
    x = BatchNormalization()(x)
    x = fire(x, 16, 64)
    x = BatchNormalization()(x)
    x = fire(x, 32, 128)
    x = BatchNormalization()(x)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    
  
    x = fire(x, 32, 128)
    x = BatchNormalization()(x)
    x = fire(x, 48, 192)
    x = BatchNormalization()(x)
    x = fire(x, 48, 192)
    x = BatchNormalization()(x)
    x = fire(x, 64, 256)
    x = BatchNormalization()(x)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    output2 = GlobalMaxPool2D()(x)
    model = Model(input,output2)
    return model


def action_model(shape=(5, 60, 60, 3), nbout=2):
    # Create our convnet with (60, 60, 3) input shape
    mobilenet= mobile_net(shape[1:]) 
    squeeze_net= Squeeze_net(shape[1:]) 
    # then create our final model
    input = Input(shape)
    x1 = TimeDistributed(mobilenet)(input)
    x2 = TimeDistributed(squeeze_net)(input)
    z  = Concatenate()([x1, x2])
    x  = GRU(64)(z)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    x  = BatchNormalization()(x)
    output  = Dense(nbout, activation='sigmoid')(x)
    model = Model(input,output)
    return model


INSHAPE=(NBFRAME,) + SIZE + (CHANNELS,) 
model = action_model(INSHAPE, len(classes))
optimizer = keras.optimizers.Adamax(0.01)
model.compile(
    optimizer,
    'categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
EPOCHS=18
z=model.fit(
    train,
    validation_data=valid,
    shuffle = 1, 
    verbose=1,
    epochs=EPOCHS,)

