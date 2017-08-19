
from __future__ import print_function
from random import shuffle
import glob
shuffle_data = True  # shuffle the addresses before saving


import numpy as np
import os
import glob
import cv2
import math
import matplotlib.pyplot as plt
import time
import re
import sys
import tensorflow as tf


from sklearn.cross_validation import train_test_split
from sklearn import preprocessing



img_w = 28
img_h = 28
img_d = 1

# image_filenames = glob.glob("C:/git/melanoma_training/*.jpg")
# print(image_filenames[0:2])


def get_im(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    if img_d == 1:
        img = cv2.imread(addr,0)
    else:
        img = cv2.imread(addr)

    img = cv2.resize(img, (img_w,img_h), interpolation=cv2.INTER_CUBIC)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img


def load_train():
    X_train = []
    X_train_label = []
    print('Read train images and labels')
    files = glob.glob('C:/git/Melanoma_training/*.jpg')
    for fl in files:
        flbase = os.path.basename(fl)
        flbase = os.path.splitext(flbase)[0]
        img_data = get_im(fl)
        #img=image_to_feature_vector(img,(128,128))
        flbase = re.findall('\d+', flbase)
        # print('label de la imagen',flbase)
        # print('imagen test no. :', fl)
        # plt.imshow(img_data, cmap = 'Greys', interpolation = 'None')
        # plt.show()

        X_train_label.append(flbase[0])
        
        X_train.append(img_data)
    #Y_train = np.loadtxt("Training_GroundTruth.csv", delimiter=",", dtype=None)
    Y_train_csv = np.genfromtxt("Training_GroundTruth.csv", delimiter=",", dtype=None)

    Y_train = []
    for j in Y_train_csv:
        Y_train.append(j[1])

    return X_train, Y_train, X_train_label


def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    # print('hola',addr)
    # print('aqui')
    # print(str(addr))
    # print(addr[0])
    # print(map(str,addr[0]))
    addr_final = 'C:/git/Melanoma_training/ISIC_' + addr + '.jpg' 
    img = cv2.imread(addr_final, 0)
    # img = cv2.resize(img, (img_w,img_h), interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, (img_w,img_h))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# def normalize_train_data():
#     train_data, Y_train_data, train_data_label = load_train()
#     train_data = np.array(train_data,  dtype=np.uint8)
#     train_data = train_data.reshape(train_data.shape[0], img_w, img_h, img_d)
#     train_data = train_data.astype('float32')
#     train_data = train_data/255
#     #Y_train_data = orderY(train_data_label, Y_train_data)
#     Y_train_data = np_utils.to_categorical(Y_train_data)
#     num_classes = Y_train_data.shape[1]
#     return train_data, Y_train_data, num_classes
# X1, Y1, classes = normalize_train_data()

X1, Y1, L1 = load_train()


Y1 = Y1[0:50]


if shuffle_data:
    c = list(zip(X1, Y1, L1))
    shuffle(c)
    X1, Y1, L1 = zip(*c)

train_X = X1[0:int(0.8*len(X1))]
train_Y = Y1[0:int(0.8*len(Y1))]
train_L = L1[0:int(0.8*len(L1))]

valid_X = X1[int(0.8*len(X1)):]
valid_Y = Y1[int(0.8*len(Y1)):]
valid_L = L1[int(0.8*len(L1)):]

train_filename = 'train.tfrecords'  # address to save the TFRecords file
# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)

print('Writing variables to TFRecords file...')
#print(train_L.shape)
for i in range(len(train_X)):
    # print how many images are saved every 1000 images
    if not i % 1:
        # print('Train data: {}/{}'.format(i, len(train_L), flush=True)
        print('Train data ', '(', i, '): ', train_L[i])
        #sys.stdout.flush()
    # Load the image
    # img = load_image(train_L[i])

    # if (i == 0):
    #     img = img.astype(np.uint8)
    #     plt.imshow(img, cmap=plt.get_cmap('gray'))
    #     plt.show()
    img = load_image(train_L[i])
    label = train_Y[i]
    name = train_L[i]

    # Create a feature
    feature = {'train/label': _int64_feature(label),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring())),
               'train/name': _bytes_feature(tf.compat.as_bytes(name))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()
#sys.stdout.flush()
