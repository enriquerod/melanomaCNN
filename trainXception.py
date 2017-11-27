import pandas as pd
import numpy as np
import logging
import argparse
import os
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import np_utils

from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import ReduceLROnPlateau


from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

from models import tiny_XCEPTION
from scipy.io import loadmat
# from utils import load_data


def load_data(mat_path):
    d = loadmat(mat_path)
    return d["train"], d["y"]


def normalize_train_data(image, y_Class, n_Class, img_w, img_h, img_d):
    # data_x = image.reshape(image.shape[0], img_w, img_h, img_d)
    image = image.reshape(image.shape[0], img_w, img_h, img_d)
    data_x = image.astype('float32')
    data_x = data_x/255
    data_y = np_utils.to_categorical(y_Class, n_Class)

    return data_x, data_y

# dataset_path = "C:/git/dataset.mat"
# img_w = 224
# img_h = 224
# img_d = 3
# n_Class = 2
# patience = 30
dataset_path = "C:/git priv/train/train_db_gray.mat"
# dataset_path = "train/train_db_gray.mat"
img_w = 256
img_h = 256
img_d = 1
n_Class = 2
patience = 30


print("Loading dataset...")
# X, y, X_val, y_val = load_data(dataset_path)
X, y = load_data(dataset_path)
# X_data, y_data_g, y_data_a = normalize_train_data(image, gender, ageClass)
# print("Task Done. Lenght Dataset:", len(X_data))
# X, X_val, y, y_val = train_test_split(X_data, y_data_a, test_size=.10)
print('data loaded')
X_train, y_train = normalize_train_data(X, y, n_Class,  img_w, img_h, img_d)
print('data normalized')
# print("Task Done. Lenght Dataset:", len(X), "and:", len(X_val))
# print("Tamano:", y.shape, "and", y_val.shape)

model = tiny_XCEPTION((img_h, img_w, img_d), n_Class, l2_regularization=0.01)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# checkpoint
filepath="pesos/weights-{epoch:02d}-{val_acc:.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
# reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience), verbose=1)
csv_logger = CSVLogger('history_class.csv')
callbacks_list = [checkpoint, csv_logger]

# Fit the model
print('Now, Training the CNN ...')
# graph = plot_model(model, to_file='model.png', show_shapes=True)
#1
# model.fit(X, y, validation_data=(X_val, y_val), nb_epoch=100, batch_size=600, verbose=2)
model.fit(X_train, y_train, nb_epoch=30, batch_size=60, verbose=1, callbacks=callbacks_list)
# history = model.fit(X, y, validation_data=(X_val, y_val), nb_epoch=30, batch_size=30, verbose=1, callbacks=callbacks_list)




























