###
# Script to create a dataset for training/validation for classification task
# 1- masking 
# 2- save on .mat
#
#
###

import numpy as np
import cv2
import scipy.io
import argparse
from tqdm import tqdm
# from utils import get_meta
import os
# from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import img_to_array, load_img
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv


def masking(img_data, img_mask):
    # print(img_data.shape, img_mask.shape)
    # plt.imshow(img_mask)
    # plt.show()
    for px in range(img_mask.shape[0]):
        for py in range(img_mask.shape[1]):
            if img_mask[px,py] == 0:
                img_data[px,py] = 0

    # plt.imshow(img_data)
    # plt.show()
    return(img_data)



def get_im(path, img_w, img_h, img_d):

    # Load as grayscale
    if img_d == 1:
        img_o = cv2.imread(path)
        img_o = cv2.resize(img_o, (img_w, img_h))
        img = cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY)
        # img = img.reshape(1, img_w, img_h, img_d)
    else:
        img_o = cv2.imread(path)
        img_o = cv2.resize(img_o, (img_w, img_h))
        img = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
        # img = img.reshape(1, img_w, img_h, img_d)
    return img, img_o

def normalize_data(img_data, img_w, img_h, img_d):
    # print("Normalizing data...")
    # img_data = img_data.reshape(1, img_w, img_h, img_d)
    train_data = np.array(img_data,  dtype=np.uint8)
    return train_data

# output_path = "img_dataset/" + str(fol) + "/" + str(fol)+'_aug_{}.png'

# path_im = "C:/Users/EHO085/Desktop/data_skin/segmentation/im_dataset"
# path_mk = "C:/Users/EHO085/Desktop/data_skin/segmentation/mk_dataset"

# if not os.path.exists(path_im):
#     os.makedirs(path_im)

# if not os.path.exists(path_mk):
#     os.makedirs(path_mk)

img_w = 256
img_h = 256
img_d = 1


path_data = "C:/Users/EHO085/Desktop/data_skin/ISIC-2017_Training_Data"
path_val = "C:/Users/EHO085/Desktop/data_skin/ISIC-2017_Validation_Data"

path_mask_data = "C:/Users/EHO085/Desktop/data_skin/ISIC-2017_Training_Part1_GroundTruth"
path_mask_val = "C:/Users/EHO085/Desktop/data_skin/ISIC-2017_Validation_Part1_GroundTruth"

path_data = "C:/Users/EHO085/Desktop/data_skin/ISIC-2017_Training_Data"
path_val = "C:/Users/EHO085/Desktop/data_skin/ISIC-2017_Validation_Data"

path_gt_data = "C:/Users/EHO085/Desktop/data_skin/ISIC-2017_Training_Part3_GroundTruth.csv"
path_gt_val = "C:/Users/EHO085/Desktop/data_skin/ISIC-2017_Validation_Part3_GroundTruth.csv"


def createDS(path_label, path_mask, img_w, img_h, img_d, output_path):
    x_data = []
    y_data = []
    label_csv = np.genfromtxt(path_label, delimiter=",", dtype=None)
    label = label_csv[1:,0]
    gt = label_csv[1:,1]

    print("Dataset length:", len(label))

    print('Read images...')

    for (files, y) in tqdm(zip(label, gt)):
        fl = path_data + '/' + files.decode("utf-8")  + '.jpg'
        X1, X1_o = get_im(fl, img_w, img_h, img_d)
        fl = path_mask + '/' + files.decode("utf-8") + '_segmentation.png'
        X2, X2_o = get_im(fl, img_w, img_h, 1)
        # print(X1.shape, X2.shape)
        data = masking(X1, X2)
        # plt.imshow(data)
        # plt.show()

        im_data = normalize_data(data, img_w, img_h, img_d)
        y = float(y)
        y = int(y)
        x_data.append(im_data)
        # x_data.append(data)
        y_data.append(y)
    
    print(len(x_data), len(y_data))
    output = {"train": x_data, "y": y_data}
    scipy.io.savemat(output_path, output)
    print("Task DONE, File .mat created")
        

output_path = 'train_db_gray.mat'
createDS(path_gt_data, path_mask_data, img_w, img_h, img_d, output_path)

# print('Read train images and masks...')
# new_train_csv = []
# # files = glob.glob('C:/git/Melanoma_training1/*.jpg')
# # for files in tqdmrain_label[0:10]):
# for files in tqdm(train_label):
#     fl = path_data + '/' + files.decode("utf-8")  + '.jpg'
#     X1, X1_o = get_im(fl, img_w, img_h, img_d)
#     fl = path_mask + '/' + files.decode("utf-8") + '_segmentation.png'
#     X2, X2_o = get_im(fl, img_w, img_h, 1)

#     # new_image = cv2.cvtColor(X1, cv2.COLOR_BGR2RGB)
#     cv2.imwrite(path_im + "/" + files.decode("utf-8") + ".png", X1_o)
#     cv2.imwrite(path_mk + "/" + files.decode("utf-8") + "_segmentation.png", X2_o)
#     new_train_csv.append(files.decode("utf-8"))

#     out_im = path_im + "/" + files.decode("utf-8")  + "_aug{}.png"
#     out_mk = path_mk + "/" + files.decode("utf-8")  + "_aug{}_segmentation.png"


#     data, mask = normalize_train_data(path_label, path_mask, path_data, img_w, img_h, img_d)

#     print("Reading Images finished!!")
#     print(len(data), len(mask))

#     output = {"train": data, "mask": mask}
#     scipy.io.savemat(output_path, output)
#     print("Task DONE, File .mat created")