from sklearn.cross_validation import train_test_split
import numpy as np
import os
import glob
import cv2
import math
import matplotlib.pyplot as plt
import time
import re
from keras.utils import np_utils

import scipy.io
import argparse
# from tqdm import tqdmclose
# from utils import get_meta


img_w = 224
img_h = 224
img_d = 3
output_path = 'C:/git/melanomaCNN/dataset.mat'

def get_im(path):
    # Load as grayscale
    if img_d == 1:
        img = cv2.imread(path,1)
    else:
        img = cv2.imread(path)
    img = cv2.resize(img, (img_w,img_h))
    return img


def load_train():
    X_train = []
    X_train_label = []
    print('Read train images and labels')
    files = glob.glob('C:/git/Melanoma_training1/*.jpg')
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
        X_train_label.append(flbase)
        X_train.append(img_data)
    #Y_train = np.loadtxt("Training_GroundTruth.csv", delimiter=",", dtype=None)
    Y_train_csv = np.genfromtxt("Training_GroundTruth.csv", delimiter=",", dtype=None)

    # print(Y_train)
    # Y_aqui1 = Y_train[0]
    # print(Y_aqui1)
    # Y_aqui1 = Y_aqui1[1]
    # print(Y_aqui1)

    Y_train = []
    for j in Y_train_csv:
        Y_train.append(j[1])


    #X_train_label = list(map(int, X_train_label))
    return X_train, Y_train, X_train_label

def load_test():
    X_test1 = []
    X_test1_label = []
    print('Read test images and labels')
    files = glob.glob('C:/git/Melanoma_testing1/*.jpg')
    for fl in files:
        flbase = os.path.basename(fl)
        flbase = os.path.splitext(flbase)[0]
        img_data = get_im(fl)
        flbase = re.findall('\d+', flbase)
        # print('label de la imagen',flbase)
        # print('imagen test no. :', fl)
        # plt.imshow(img_data, cmap = 'Greys', interpolation = 'None')
        # plt.show()
        #img=image_to_feature_vector(img,(128,128))
        X_test1_label.append(flbase)
        X_test1.append(img_data)
    Y_test1_csv = np.genfromtxt("Test_GroundTruth.csv", delimiter=",", dtype=None)
    Y_test1 = []
    for j in Y_test1_csv:
        Y_test1.append(j[1])

    #X_test1_label = list(map(int, X_test1_label))
    #X_test1_label.sort()
    return X_test1, Y_test1, X_test1_label

def normalize_train_data():
    train_data, Y_train_data, train_data_label = load_train()
    train_data = np.array(train_data,  dtype=np.uint8)
    train_data = train_data.reshape(train_data.shape[0], img_w, img_h, img_d)
    # train_data = train_data.reshape(train_data.shape[0], img_d, img_w, img_h)
    train_data = train_data.astype('float32')
    train_data = train_data/255
    #Y_train_data = orderY(train_data_label, Y_train_data)
    Y_train_data = np_utils.to_categorical(Y_train_data)
    num_classes = Y_train_data.shape[1]
    return train_data, Y_train_data, num_classes

def normalize_test_data():
    test_data, Y_test_data , test_data_label= load_test()
    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.reshape(test_data.shape[0], img_w, img_h, img_d)
    test_data = test_data.astype('float32')
    #X_test = preprocessing.scale(X_test)
    test_data = test_data/255

    # = orderY(test_data_label, Y_test_data)

    Y_test_data = np_utils.to_categorical(Y_test_data)
    return test_data, Y_test_data



X1, Y1, classes = normalize_train_data()
X_test, Y_test = normalize_test_data()


X, X_val, Y, Y_val = train_test_split(X1, Y1, test_size=.30)

output = {"X": np.array(X), "Y": Y, "X_val": np.array(X_val) ,"Y_val": Y_val}
scipy.io.savemat(output_path, output)
print('Dataset .mat created')
