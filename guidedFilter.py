import numpy as np
import os
import glob
import cv2
import math
import matplotlib.pyplot as plt
import time
import re

from imagerie import *
from fusion import *
from scipy.misc import toimage, imsave
from scipy.ndimage.filters import laplace
from scipy.ndimage import uniform_filter, gaussian_filter

print('Reading images...')
# img1 = cv2.imread('22.jpg')
# img2 = cv2.imread('22.jpg')
# img3 = cv2.imread('22.jpg')

img1 = imread("8.jpg")
img2 = imread("9.jpg")
img3 = imread("22.jpg")

# plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
# plt.show()
print('Appliying Filter...')
fused = cGFF([img1, img2])

imwrite("fused.jpg", fused)
print('Task Finished')




# def extract_maps(im, average_filter_size=31, sigma_r=5):
#     base_layer = uniform_filter(im, size=average_filter_size)
#     detail_layer = im - base_layer
#     if len(im.shape) >= 3:
#         imgray = rgb2gray(im)
#     else:
#         imgray = im.astype(np.float64) - 0.0001
#     saliency = gaussian_filter(abs(laplace(imgray)), sigma_r)
#     return base_layer, detail_layer, saliency

# def GFF(images,
#         average_filter_size = 31,
#         sigma_r=5,
#         r_base=5,
#         epsilon_base=2,
#         r_detail=5,
#         epsilon_detail=0.1):
    
#     base_layers, detail_layers, saliencies = zip(*[extract_maps(im, average_filter_size, sigma_r) for im in images])
#     saliency_idx = np.argmax(saliencies, axis=0)
    
#     masks = [saliency_idx == i for i in range(len(images))]
    
#     base_weights = [guided_filtering(mask, im / 255, size=r_base, epsilon=epsilon_base) for im, mask in zip(images, masks)]
#     detail_weights = [guided_filtering(mask, im / 255, size=r_detail, epsilon=epsilon_detail) for im, mask in zip(images, masks)]
    
#     # a bit hacky
#     if len(images[0].shape) >= 3:
#         base_weights = [w[:,:,None] for w in base_weights]
#         detail_weights = [w[:,:,None] for w in detail_weights]
    
#     base = sum([base * weight for base, weight in zip(base_layers, base_weights)]) / sum(base_weights)
#     detail = sum([detail * weight for detail, weight in zip(detail_layers, base_weights)]) / sum(base_weights)
    
#     return detail + base

# final = GFF([img1, img2])
# print(final.shape)
# cv2.imshow("closeLAB", final)

# # plt.imshow(final)
# # plt.show()

# plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
# plt.show()
# cv2.waitKey(0)