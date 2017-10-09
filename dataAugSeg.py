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



###########################################################################################################

##########################################################################################




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


path_label = "C:/Users/EHO085/Desktop/data skin/ISIC-2017_Validation_Data/ISIC-2017_Validation_Data_metadata.csv"



img_w = 256
img_h = 256
img_d = 3
path_data = "C:/Users/EHO085/Desktop/data skin/ISIC-2017_Training_Data/ISIC_0014682.jpg"
path_mask = "C:/Users/EHO085/Desktop/data skin/ISIC-2017_Training_Part1_GroundTruth/ISIC_0014682_segmentation.png"

X1 = cv2.imread(path_data)
X1= cv2.cvtColor(X1, cv2.COLOR_BGR2RGB)
X1 = cv2.resize(X1, (img_w, img_h))
X1 = X1.reshape(1, img_w, img_h, img_d)

X2 = cv2.imread(path_mask)
X2= cv2.cvtColor(X2, cv2.COLOR_BGR2GRAY)
X2 = cv2.resize(X2, (img_w, img_h))
X2 = X2.reshape(1, img_w, img_h, 1)

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


output_path = 'img_{}.png'
output_path2 = 'mask_{}.png'
count = 5

for  i, (new_images, new_masks) in enumerate(zip(image_generator, mask_generator)):  
    # we access only first image because of batch_size=
#     new_images = data_image[0]
#     new_masks = data_image[1]
    new_image = new_images[0].reshape(img_w, img_h, img_d)
    new_mask = new_masks[0].reshape(img_w, img_h, 1)
    # new_image = array_to_img(new_images[0], sScale=True)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path.format(i + 1), new_image)
#     new_mask = cv2.cvtColor(new_mask, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path2.format(i + 1), new_mask)
    # new_image.save(output_path.format(i + 1))
    if i >= count-1:
        break

im = mpimg.imread("img_1.png")
mk = mpimg.imread("mask_1.png")
masking(im, mk)

# 
# 
# output_path = 'dog_random{}.png'
# count = 10
# print("Augmenting")
# images_flow = datagen.flow(X1, batch_size=1)
# print("Aug Finished")
# for i, new_images in enumerate(images_flow):  
#     # we access only first image because of batch_size=1
#     new_image = new_images[0].reshape(img_w, img_h, img_d)
#     # new_image = array_to_img(new_images[0], scale=True)
#     new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
#     cv2.imwrite(output_path.format(i + 1), new_image)
#     # new_image.save(output_path.format(i + 1))
#     if i >= count-1:
        # break



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