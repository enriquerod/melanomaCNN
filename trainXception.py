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
from utils import load_data



dataset_path = "C:/git/dataset.mat"
img_w = 224
img_h = 224
img_d = 3
n_Class = 2
patience = 30

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
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience), verbose=1)
csv_logger = CSVLogger('history.csv')
callbacks_list = [checkpoint, csv_logger, reduce_lr]

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


print(history.history.keys())  
plt.figure(1)  

print("Plotting Loss and Accuracy...")
# summarize history for accuracy  

plt.subplot(211)  
plt.plot(history.history['acc'])  
plt.plot(history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  

# summarize history for loss  

plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.show() 




