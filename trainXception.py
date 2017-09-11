import pandas as pd
import numpy as np
import logging
import argparse
import os
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import np_utils

from keras.callbacks import ModelCheckpoint

from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

from models import tiny_XCEPTION
from utils import load_data

# def normalize_train_data(image, mClass):
#     data_x = image.reshape(image.shape[0], img_w, img_h, img_d)
#     data_x = data_x.astype('float32')
#     data_x = data_x/255
#     data_y = np_utils.to_categorical(gender, 2)
#     return data_x, data_y


# dataset_path = "C:/Users/EHO085/Desktop/Face Group Age/data/wiki_db.mat"
dataset_path = "C:/git/dataset.mat"
img_w = 224
img_h = 224
img_d = 3
n_Class = 2

print("Loading dataset...")
# image, gender, ageClass, _, _, image_size, _ = load_data(dataset_path)
X, y, X_val, y_val = load_data(dataset_path)
# X_data, y_data_g, y_data_a = normalize_train_data(image, gender, ageClass)
# print("Task Done. Lenght Dataset:", len(X_data))
# X, X_val, y, y_val = train_test_split(X_data, y_data_a, test_size=.10)
print("Task Done. Lenght Dataset:", len(X), "and:", len(X_val))
print("Tamano:", y.shape, "and", y_val.shape)

model = tiny_XCEPTION((img_h, img_w, img_d), n_Class, l2_regularization=0.01)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# checkpoint
filepath="weights-{epoch:02d}-{val_acc:.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# Fit the model
print('Now, Training the CNN ...')
# graph = plot_model(model, to_file='model.png', show_shapes=True)
#1
# model.fit(X, y, validation_data=(X_val, y_val), nb_epoch=100, batch_size=600, verbose=2)
history = model.fit(X, y, validation_data=(X_val, y_val), nb_epoch=15, batch_size=30, verbose=2, callbacks=callbacks_list)
model.save_weights("model.h5")
print('Training Finished and Weights saved!!!')
np.save('history.npy', history) 
print('History saved!!!')



