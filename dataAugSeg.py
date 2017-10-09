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

# img_w = 128
# img_h = 128
# img_d = 3

# img_w = 64
# img_h = 64
# img_d = 3

# X1 = cv2.imread("prueba.png")
# X1= cv2.cvtColor(X1, cv2.COLOR_BGR2RGB)
# X1 = cv2.resize(X1, (img_w, img_h))
# X1 = X1.reshape(1, img_w, img_h, img_d)
# Y1 = 1


# # plt.figure(1)
# # # create a grid of 3x3 images
# # for i in range(0, 9):
# #     plt.subplot(330 + 1 + i)
    
# #     if img_d == 1:
# #         img_aux = X1[i].reshape(img_w, img_h)
# #         plt.imshow(img_aux, cmap=plt.get_cmap('gray'))
# #         #plt.imshow(cv2.cvtColor(img_aux, cv2.COLOR_BGR2GRAY))
        
# #     else:
# #         img_aux = X1[i].reshape(img_w, img_h, img_d)
# #         plt.imshow(cv2.cvtColor(img_aux, cv2.COLOR_BGR2RGB))
# # # show the plot

# print('Augmenting ...')

# datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')


# # datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# #datagen = ImageDataGenerator(rotation_range=90)
# #datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

# # shift = 0.2
# # datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)

# #datagen = ImageDataGenerator(zca_whitening=True)

# # fit parameters from data
# # datagen.fit(X1)
# # configure batch size and retrieve one batch of images
# # plt.hold(True)
# # plt.figure(2)
# path_f= "C:/Users/EHO085/Desktop/Datasets/Product Store/inVitro"
# path_v = "C:/Users/EHO085/Desktop/Datasets/Product Store/inVitro"
# path_label = "situ.csv"
# name_f = "vitro"
# train_label = np.genfromtxt(path_label, delimiter=",", dtype=None)
# name_f = "{}.csv".format(name_f)
# # Search for all the images inside of the *folder
# fol_v = os.listdir(path_f)

# # os.chdir(path_v)
# # f = (file for file in glob.glob(path))
# #     # print(file)
# # print(f)
# # print(len(f))

# fol_v = list(map(int, fol_v))
# fol_v = sorted(fol_v)
# fol_v = list(map(str, fol_v))

# fol_v_p_i =[]
# fol_v_p = []
# samples = []
# # for fol in fol_v:
# for fol in train_label:

#         path = path_v + "/" +str(fol) + "/web/PNG/*.png"
#         fol_v_p_i = []
#         i = 0
#         print(path)
#         if not os.path.exists("dataset/" + str(fol)):
#                 os.makedirs("dataset/" + str(fol))



#         for files in glob.glob(path):

#                 i+=1
#                 fol_v_p_i.append(files)
                
                
#                 k, _ = os.path.splitext(os.path.basename(files))
       
#                 X1 = cv2.imread(files)
                
#                 path_s = "dataset/"+ str(fol) + "/" + str(fol) + k +  ".png"
                
#                 cv2.imwrite(path_s, X1)
#                 X1= cv2.cvtColor(X1, cv2.COLOR_BGR2RGB)
#                 print("AQUI2")
#                 X1 = cv2.resize(X1, (img_w, img_h))
#                 X1 = X1.reshape(1, img_w, img_h, img_d)


#                 output_path = "dataset/" + str(fol) + "/" + str(fol)+ k +'_aug_{}.png'
#                 count = 50
#                 print("Augmenting")
#                 images_flow = datagen.flow(X1, batch_size=1)
#                 print("Aug Finished")
#                 for i, new_images in enumerate(images_flow):  
#                         # we access only first image because of batch_size=1
#                         new_image = new_images[0].reshape(img_w, img_h, img_d)
#                         # new_image = array_to_img(new_images[0], scale=True)
#                         new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
#                         cv2.imwrite(output_path.format(i + 1), new_image)
#                         # new_image.save(output_path.format(i + 1))
#                         if i >= count-1:
#                                 break
                
#         # print(fol_v_p_i)
#         # print(i)
#         # input()
#         samples.append(i)
#         fol_v_p.append(fol_v_p_i)


##########################################################################################

###########################################################################################################

##########################################################################################




path_label = "C:/Users/EHO085/Desktop/data skin/ISIC-2017_Validation_Data/ISIC-2017_Validation_Data_metadata.csv"
path_mask = "C:/Users/EHO085/Desktop/data skin/ISIC-2017_Validation_Part1_GroundTruth"
path_data = "C:/Users/EHO085/Desktop/data skin/ISIC-2017_Validation_Data"


img_w = 256
img_h = 256
img_d = 3
path_data = "C:/Users/EHO085/Desktop/data skin/ISIC-2017_Training_Data/ISIC_0000000.jpg"
path_mask = "C:/Users/EHO085/Desktop/data skin/ISIC-2017_Training_Part1_GroundTruth/ISIC_0000000_segmentation.png"

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
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
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
    # new_image = array_to_img(new_images[0], scale=True)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path.format(i + 1), new_image)
#     new_mask = cv2.cvtColor(new_mask, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path2.format(i + 1), new_mask)
    # new_image.save(output_path.format(i + 1))
    if i >= count-1:
        break





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