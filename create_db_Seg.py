import numpy as np
import cv2
import scipy.io
import argparse
from tqdm import tqdm
# from utils import get_meta

import matplotlib.pyplot as plt
import os
import glob

#BUILD DATASET IN .MAT FORMAT FOR 
#SEGMENTATION
#Inputs: images and masks

def get_args():
    parser = argparse.ArgumentParser(description="This script build a .mat dataset ",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="path to output database mat file")
    # parser.add_argument("--db", type=str, default="wiki",
    #                     help="dataset; wiki or imdb")
    # parser.add_argument("--db", type=str, default="imdb",
    #                     help="dataset; wiki or imdb")
    # parser.add_argument("--img_size", type=int, default=64,
    #                     help="output image size")
    # parser.add_argument("--min_score", type=float, default=2,
    #                     help="minimum face_score")
    args = parser.parse_args()
    return args


def get_im(path, img_w, img_h, img_d):

    # Load as grayscale
    if img_d == 1:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_w,img_h))
    
    return img

def masking(img_data, img_mask):
    for px in range(img_mask.shape[0]):
        for py in range(img_mask.shape[1]):
            if img_mask[px,py] == 0:
                img_data[px,py] = 255

    plt.imshow(img_data, cmap='gray')
    plt.show()



def load_train(path_label, path_mask, path_data, img_w, img_h, img_d):
    print("Reading Labels...")
    train_label = np.genfromtxt(path_label, delimiter=",", dtype=None)
    train_label = train_label[1:,0]
    print("Dataset length:", len(train_label))

    train_data= []
    train_mask = []
    print('Read train images and masks...')
    # files = glob.glob('C:/git/Melanoma_training1/*.jpg')
    # for files in tqdm(train_label[0:100]):
    for files in tqdm(train_label):
        fl = path_data + '/' + files.decode("utf-8")  + '.jpg'
        img_data = get_im(fl, img_w, img_h, img_d)

        # img1 = cv2.imread(fl)
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        fl = path_mask + '/' + files.decode("utf-8") + '_segmentation.png'
        img_mask = get_im(fl, img_w, img_h, img_d)
        
        # img2 = cv2.imread(fl)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # masking(img_data, img_mask)
        # masking(img1, img2)


        train_data.append(img_data)
        train_mask.append(img_mask)

    return train_label, train_mask, train_data



def normalize_train_data(path_label, path_mask, path_data, img_w, img_h, img_d):
    train_label, train_mask, train_data = load_train(path_label, path_mask, path_data, img_w, img_h, img_d)
    
    print("Normalizing data...")
    train_data = np.array(train_data,  dtype=np.uint8)
    # train_data = train_data.reshape(train_data.shape[0], img_w, img_h, img_d)
    # # train_data = train_data.reshape(train_data.shape[0], img_d, img_w, img_h)
    # train_data = train_data.astype('float32')
    # train_data = train_data/255

    train_mask = np.array(train_mask,  dtype=np.uint8)
    # train_mask = train_mask.reshape(train_mask.shape[0], img_w, img_h, img_d)
    # # train_data = train_data.reshape(train_data.shape[0], img_d, img_w, img_h)
    # train_mask = train_mask.astype('float32')
    # train_mask = train_mask/255


    return train_data, train_mask



def main():
    args = get_args()
    output_path = args.output
    db = args.db
    img_size = args.img_size
    min_score = args.min_score

    img_w = 512
    img_h = 512
    img_d = 1
    # output_path = 'dataset.mat'
    output_path = args.output

    path_label = "data/ISIC-2017_Training_Part3_GroundTruth.csv"
    path_mask = "C:/Users/EHO085/Desktop/data skin/ISIC-2017_Training_Part1_GroundTruth"
    path_data = "C:/Users/EHO085/Desktop/data skin/ISIC-2017_Training_Data"

    # mat_path = root_path + "{}.mat".format(db)

    data, mask = normalize_train_data(path_label, path_mask, path_data, img_w, img_h, img_d)

    print("Reading Images finished!!")


    output = {"train": data, "mask": mask}
    scipy.io.savemat(output_path, output)
    print("Task DONE, File .mat created")

if __name__ == '__main__':
    main()
