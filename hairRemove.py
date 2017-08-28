import numpy as np
import os
import glob
import cv2
import math
import matplotlib.pyplot as plt
import time
import re
from scipy import ndimage

img = cv2.imread('3.jpg')

kernel = np.ones((5,5),np.uint8)
kernel2 = np.ones((3,3),np.uint8)


# cv2.imshow("original", img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


imgLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# cv2.imshow("original LAB", imgLAB)
plt.imshow(cv2.cvtColor(imgLAB, cv2.COLOR_BGR2RGB))
plt.show()
imgLAB1 = imgLAB[:,:,0]
plt.imshow(imgLAB1, cmap = 'gray')
plt.show()

closeLAB = cv2.morphologyEx(imgLAB, cv2.MORPH_CLOSE, kernel)
print(imgLAB.shape, closeLAB.shape)
imgLAB2 = closeLAB[:,:,0]
print(imgLAB2.shape)
plt.imshow(imgLAB2, cmap = 'gray')
plt.show()

finalLAB = imgLAB2 - imgLAB1
plt.imshow(finalLAB, cmap = 'gray')
plt.show()

ret,thresh1 = cv2.threshold(finalLAB,20,255,cv2.THRESH_BINARY)
plt.imshow(thresh1, cmap = 'gray')
plt.show()

thresh2 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel2)
plt.imshow(thresh2, cmap = 'gray')
plt.show()
print('hola',thresh2.shape)
print(img.shape)
m = thresh2.shape[0]
n = thresh2.shape[1]
print(n)
print(m)
print(img.dtype)

# a = thresh2 > 0
# print(a)
# x = np.argwhere(a)
# print(x)
# x[:,[0,1]] = x[:,[1,0]]

# for k,i in enumerate(x):
#     print(k)
#     img.itemset((i[0],i[1],0),0)
#     img.itemset((i[0],i[1],1),0)
#     img.itemset((i[0],i[1],2),0)


# print(x)
# print(x.shape)
# print(img.shape)
# img[x] = [0,0,0]


for i in range(thresh2.shape[0]):
    for j in range(thresh2.shape[1]):
        if (thresh2[i,j] > 0):
            # img[i,j] = [0,0,0]
            img.itemset((i,j,0),255)
            img.itemset((i,j,1),255)
            img.itemset((i,j,2),255)



plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

img = ndimage.median_filter(img, 3)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# cv2.imshow("closeLAB", closeLAB)
# cv2.waitKey(0)

# img = cv2.imread(files[0], 0)
# print('imprimir segunda')
# plt.imshow(img, cmap = 'Greys', interpolation = 'None')
# plt.show()