import pandas as pd
import logging
import argparse
import os
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import np_utils
from utils import mk_dir, load_data
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from keras.utils.vis_utils import plot_model
from keras.models import model_from_json

import cv2
import numpy as np
import matplotlib.pyplot as plt

from models import tiny_XCEPTION



def get_args():
    parser = argparse.ArgumentParser(description="This script runs a cnn classfication "
                                                 "for product recognition.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--im_path", "-i_p", type=str, required=True,
                        help="path to output database mat file")
    # # parser.add_argument("--db", type=str, default="wiki",
    # #                     help="dataset; wiki or imdb")
    # parser.add_argument("--db", type=str, default="imdb",
    #                     help="dataset; wiki or imdb")
    # parser.add_argument("--img_size", type=int, default=64,
    #                     help="output image size")
    # parser.add_argument("--min_score", type=float, default=2,
    #                     help="minimum face_score")
    args = parser.parse_args()
    return args



# def classes(i_class):
#     if i_class == 12:
#         o_class = 'Sprite'
#     if i_class == 19:
#         o_class = 'Coca-Cola'
#     if i_class == 20:
#         o_class = 'Emperador'
#     if i_class == 21:
#         o_class = 'Hony Bran'
    
#     return o_class






#   480 x 640 pixeles
#   area of 
#
#
#


def main():

    # args = get_args()
    # image_path = args.im_path

    img_w = 64
    img_h = 64
    img_d = 1
    x1 = 150
    x2 = 400
    y1 = 50
    y2 = 400

    n_Class = 20
    # model = tiny_XCEPTION((img_h, img_w, img_d), 6, l2_regularization=0.01)
    model = tiny_XCEPTION((img_h, img_w, img_d), n_Class, l2_regularization=0.01)
    model.load_weights("weights-23.hdf5")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Weights model loaded from disk")

    # detection_model_path = 'haarcascade_frontalface_default.xml'
    # face_detection = cv2.CascadeClassifier(detection_model_path)

    video_capture = cv2.VideoCapture(0)
    # image_test = cv2.imread('video45.png')
    while True:
        image_test = video_capture.read()[1]
        print("Dimensiones: ", image_test.shape)
        gray_image = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)

        rgb_image = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)

        rgb_image = gray_image
        cv2.rectangle(image_test, (x1, y1), (x2, y2), (255,0,0), 2)
        
        # rgb_image = image_test

        # bgr_image = cv2.imread("prueba4.jpg")
        # gray_image = cv2.imread("prueba4.jpg", 0)
        rgb_image = rgb_image[y1:y2, x1:x2]

        cv2.imshow("croped", rgb_image)

        image = cv2.resize(rgb_image, (img_w, img_h))
        data_x = image.reshape(1, img_w, img_h, img_d)
        data_x = data_x.astype('float32')
        data_x = data_x/255
        preds = model.predict(data_x)
        # print(preds)

        print('Clase: ',np.argmax(preds,axis=1)+1)