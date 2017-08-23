import numpy as np
import os
import glob
import cv2
import math
import matplotlib.pyplot as plt
import time
import re

img = cv2.imread('2.jpg')

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
# cv2.imshow("closeLAB", closeLAB)
# cv2.waitKey(0)

# img = cv2.imread(files[0], 0)
# print('imprimir segunda')
# plt.imshow(img, cmap = 'Greys', interpolation = 'None')
# plt.show()