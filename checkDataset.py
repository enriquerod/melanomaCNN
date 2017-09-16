import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils import get_meta
from scipy.io import loadmat

def load_data(mat_path):
    d = loadmat(mat_path)

    return d["X"], d["Y"], d["X_val"],d["Y_val"]



mat_path = "C:/git/melanomaCNN/dataset.mat"
X, Y, X_val, Y_val = load_data(mat_path)



print(len(X))

hist = plt.hist(Y, bins=np.arange(0, 3, 1), color='b')
plt.xlabel("Class training")
plt.show()

hist = plt.hist(Y_val, bins=np.arange(0, 3, 1), color='b')
plt.xlabel("Class validation")
plt.show()

