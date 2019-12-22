# -*- coding: utf-8 -*-
"""
CPSC-481 : AI - Final Project
Team Members: Pranav Dilip Borole (CWID: 887465383)
              Shaunak Narendra Deshpande (CWID: 887460368)
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv2
import os
import keras
import ssl
from keras import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical, np_utils
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, Activation, GlobalAveragePooling2D
from keras.optimizers import RMSprop, Adam
from keras.layers.normalization import BatchNormalization
import keras.backend as TF
from keras.models import load_model
from sklearn.utils import shuffle
from keras.applications.mobilenet import MobileNet

img_height,img_width = (224,224)
channels_number = 3

ssl._create_default_https_context = ssl._create_unverified_context

TRAIN_DIR = '../images/train/'
TEST_DIR = '../images/test/'
VAL_DIR = '../images/val/'
Animal_CLASSES = os.listdir(TRAIN_DIR)
print(Animal_CLASSES)

def get_images(animals):
    """Load files from train folder"""
    animals_dir = TRAIN_DIR+'{}'.format(animals)
    images = [animals+'/'+im for im in os.listdir(animals_dir)]
    return images

def read_image(src):
    """Read and resize individual images"""
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (img_height,img_width), interpolation=cv2.INTER_CUBIC)
    return im

files = []
y_all = []

for animals in Animal_CLASSES:
    animals_files = get_images(animals)
    files.extend(animals_files)
    
    y_animals = np.tile(animals, len(animals_files))
    y_all.extend(y_animals)
    print("{0} photos of {1}".format(len(animals_files), animals))
    
y_all = np.array(y_all)
print(y_all.shape)

y_all = LabelEncoder().fit_transform(y_all)
y_all = np_utils.to_categorical(y_all)
print(y_all.shape)

X_all = np.ndarray((len(files), img_height,img_width, channels_number), dtype=np.uint8)

for i, im in enumerate(files): 
    X_all[i] = read_image(TRAIN_DIR+im)
    if i%51 == 0: print('Processed {} of {}'.format(i, len(files)))

print(X_all.shape)

y_all[:5,:]

X_all,y_all=shuffle(X_all,y_all)
y_all[0:10,:]

X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all,test_size=0.05, random_state=42, stratify=y_all)
print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)


model = MobileNet(input_shape=(img_height,img_width,channels_number), alpha=1, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet', input_tensor=None, pooling=None, classes=2)
x = model.output
#x = Dropout(0.22)(x)
x = GlobalAveragePooling2D()(x)
#x = Dropout(0.10)(x)
#x = Dense(1024, activation='relu')(x)
#x = Dropout(0.21)(x)
#x = Dense(512, activation='relu')(x)
#x = Dropout(0.2)(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(30, activation='softmax')(x)
model = Model(inputs=model.input, outputs=predictions)
model.summary()

gen = ImageDataGenerator(width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            rotation_range=15,
            horizontal_flip=True,
            zoom_range=0.2)

train_batches = gen.flow(X_train, y_train, batch_size=44, seed=42)

adam = Adam(lr=0.000005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000001)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

#weights = ModelCheckpoint('model.h5', monitor='acc', save_best_only=True, save_weights_only=False)
#annealer = LearningRateScheduler(lambda x: 1e-3 * 1.025 ** x)

hist = model.fit(X_train, y_train, 8,validation_data=(X_valid,y_valid),
                           epochs=5, verbose=1) #, callbacks = [weights, annealer]

model.save('model1.hdf5')