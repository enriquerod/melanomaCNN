import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import imread
from scipy.misc import imresize, toimage
plt.rcParams['figure.figsize'] = (14, 8)

def rgb2gray(img):
    w = np.array([0.2989, 0.5870, 0.1140]).reshape((1, 1, -1))
    return (img * w).sum(axis=2)

def rgba2gray(img):
    w = np.array([0.2989, 0.5870, 0.1140, 0.0]).reshape((1, 1, -1))
    return (img * w).sum(axis=2)

def imshow(img):
    if len(img.shape) == 2 or img.shape[2] == 1:
        if img.dtype == np.dtype('int'):
            f = plt.imshow(img, cmap='gray', clim=(0, 255))
        else:
            f = plt.imshow(img, cmap='gray')
    else:
        f = plt.imshow(img)
    plt.show()
    return f
        
def plot_histogram(x, title=None):
    hist, bins = x
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width, color='black')
    if title is not None:
        plt.title(title)
    plt.show()
    
tr = np.transpose

def histogram_equalization(imgray):
    imhist, bins = np.histogram(imgray, bins=range(257))
    imhist = imhist / imgray.size
    imhistcum = imhist.cumsum()
    imeq = imhistcum[imgray.astype(int)] * 256
    return imeq

def add_gaussian_noise(u, sigma=1.0):
    return u + sigma * np.random.randn(*(u.shape))

def add_uniform_noise(u, amp=1.0):
    return u + np.random.uniform(low=-amp, high=+amp, size=u.shape)

def add_impulse_noise(u, p=1.0):
    mask = np.random.random(u.shape) < p
    noise = np.random.randint(256, size=u.shape)
    return u * (1 - mask) + noise * mask

def uniformly_quantify(u, k):
    step = (k - 1) / 255
    return (np.floor(u * step) + 0.5) / step

def imwrite(filename, matrix):
    toimage(matrix, cmin=0, cmax=256).save(filename)
