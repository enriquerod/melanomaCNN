import numpy as np
import cv2
import scipy.io
import argparse
from tqdm import tqdm
# from utils import get_meta
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv

##################################################################################


# Th script makes 5 images and mask from each image of the dataset


####################################################################################

def masking(img_data, img_mask):
    # print(img_data.shape, img_mask.shape)
    # plt.imshow(img_mask)
    # plt.show()
    for px in range(img_mask.shape[0]):
        for py in range(img_mask.shape[1]):
            if img_mask[px,py] == 0:
                img_data[px,py] = 0

    plt.imshow(img_data)
    plt.show()



def get_im(path, img_w, img_h, img_d):

    # Load as grayscale
    if img_d == 1:
        img_o = cv2.imread(path)
        img_o = cv2.resize(img_o, (img_w, img_h))
        img = cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY)
        img = img.reshape(1, img_w, img_h, img_d)
    else:
        img_o = cv2.imread(path)
        img_o = cv2.resize(img_o, (img_w, img_h))
        img = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
        img = img.reshape(1, img_w, img_h, img_d)
    return img, img_o


# output_path = "img_dataset/" + str(fol) + "/" + str(fol)+'_aug_{}.png'

path_im = "C:/Users/EHO085/Desktop/data_skin/segmentation/im_dataset"
path_mk = "C:/Users/EHO085/Desktop/data_skin/segmentation/mk_dataset"

if not os.path.exists(path_im):
    os.makedirs(path_im)

if not os.path.exists(path_mk):
    os.makedirs(path_mk)

img_w = 256
img_h = 256
img_d = 3


path_data = "C:/Users/EHO085/Desktop/data_skin/ISIC-2017_Training_Data"
path_mask = "C:/Users/EHO085/Desktop/data_skin/ISIC-2017_Training_Part1_GroundTruth"
path_label = "data/ISIC-2017_Training_Part3_GroundTruth.csv"

count = 5

train_csv = np.genfromtxt(path_label, delimiter=",", dtype=None)
train_label = train_csv[1:,0]
print("Dataset length:", len(train_label))


# name_f = "{}.csv".format(name_f)
# Search for all the images inside of the *folder
# fol_v = os.listdir(path_f)


# with open('labels.csv', 'w') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     wr.writerow(h)

# h = list(map(int, h))
# h = h.astype('int')
# np.savetxt(name_f, h, fmt='%i', delimiter=",")


print('Read train images and masks...')
new_train_csv = []
# files = glob.glob('C:/git/Melanoma_training1/*.jpg')
# for files in tqdmrain_label[0:10]):
for files in tqdm(train_label):
    fl = path_data + '/' + files.decode("utf-8")  + '.jpg'
    X1, X1_o = get_im(fl, img_w, img_h, img_d)
    fl = path_mask + '/' + files.decode("utf-8") + '_segmentation.png'
    X2, X2_o = get_im(fl, img_w, img_h, 1)

    # new_image = cv2.cvtColor(X1, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path_im + "/" + files.decode("utf-8") + ".png", X1_o)
    cv2.imwrite(path_mk + "/" + files.decode("utf-8") + "_segmentation.png", X2_o)
    new_train_csv.append(files.decode("utf-8"))

    out_im = path_im + "/" + files.decode("utf-8")  + "_aug{}.png"
    out_mk = path_mk + "/" + files.decode("utf-8")  + "_aug{}_segmentation.png"
    # X1 = cv2.imread(path_data)
    # X1= cv2.cvtColor(X1, cv2.COLOR_BGR2RGB)
    # X1 = cv2.resize(X1, (img_w, img_h))
    # X1 = X1.reshape(1, img_w, img_h, img_d)

    # X2 = cv2.imread(path_mask)
    # X2= cv2.cvtColor(X2, cv2.COLOR_BGR2GRAY)
    # X2 = cv2.resize(X2, (img_w, img_h))
    # X2 = X2.reshape(1, img_w, img_h, 1)

    # we create two instances with the same arguments
    data_gen_args = dict(featurewise_center=True,
                        featurewise_std_normalization=True,
                        rotation_range=45,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=0.2)

    # data_gen_args = dict(rotation_range=45,
    #                      width_shift_range=0.1,
    #                      height_shift_range=0.1,
    #                      zoom_range=0.2)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    # image_datagen.fit(images, augment=True, seed=seed)
    # mask_datagen.fit(masks, augment=True, seed=seed)

    image_generator = image_datagen.flow(X1,seed=seed)

    mask_generator = mask_datagen.flow(X2,seed=seed)

    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)


    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=2.0,
    #     epochs=50)
    #


    for  i, (new_images, new_masks) in enumerate(zip(image_generator, mask_generator)):  
        # we access only first image because of batch_size=
    #     new_images = data_image[0]
    #     new_masks = data_image[1]
        new_train_csv.append(files.decode("utf-8")  + "_aug" + str(i))
        new_image = new_images[0].reshape(img_w, img_h, img_d)
        new_mask = new_masks[0].reshape(img_w, img_h, 1)
        # new_image = array_to_img(new_images[0], sScale=True)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(out_im.format(i + 1), new_image)
    #     new_mask = cv2.cvtColor(new_mask, cv2.COLOR_BGR2RGB)
        cv2.imwrite(out_mk.format(i + 1), new_mask)
        # new_image.save(output_path.format(i + 1))
        if i >= count-1:
            break


with open('C:/Users/EHO085/Desktop/data_skin/segmentation/train_seg_aug.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for row in new_train_csv:
        wr.writerow([row])
    # wr.writerows(new_train_csv)


# im = mpimg.imread("img_1.png")
# mk = mpimg.imread("mask_1.png")
# masking(im, mk)



# os.makedirs('augmented')
# #, save_prefix='aug'
# i=0
# for batch in datagen.flow(X1, save_to_dir='augmented', save_format='png', save_prefix='prod'):
#     # for i in range(0, 9):
#     #     plt.subplot(330 + 1 + i)
#     #     img_aux = X_batch[i].reshape(img_w, img_h, img_d)
#         # if img_d == 1:
#         #     img_aux = X_batch[i].reshape(img_w, img_h)
#         #     plt.imshow(img_aux, cmap=plt.get_cmap('gray'))
#         # else:
#         #     img_aux = X_batch[i].reshape(img_w, img_h, img_d)
#         #     plt.imshow(cv2.cvtColor(img_aux, cv2.COLOR_BGR2RGB))
    
#     #plt.show()
#     i+=1
#     if i >= 30:
#         break