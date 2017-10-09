import matplotlib.pyplot as plt
import numpy as np

from scipy.io import loadmat

path_history =  "C:/Users/EHO085/Desktop/unet-master2/weights-val-gray-256/history.csv"

seg_history = np.genfromtxt(path_history, delimiter=",", dtype=None)
epoch = seg_history[1:,0]
acc = seg_history[1:,1]
val_acc = seg_history[1:,3]
loss = seg_history[1:,2]
val_loss = seg_history[1:,4]

# seg_history = seg_history.decode("utf-8")

# fl = path_data + '/' + files.decode("utf-8")  + '.jpg'
# X1 = get_im(fl, img_w, img_h, img_d)

plt.plot(epoch, acc, label='Acc')
plt.plot(epoch, val_acc, label='Val_acc') 
plt.plot(epoch, loss, label='Loss')
plt.plot(epoch, val_loss, label='Val_loss') 
plt.xlabel('Epochs')
# plt.ylabel('y label')

plt.title("Segmentation Metrics")

plt.legend()
# plt.plot([1,2,3,4], [1,4,9,16], 'ro')
# plt.axis([0, 6, 0, 20])
plt.show()


path_histo = "C:/Users/EHO085/Desktop/unet-master2/data/ISIC-2017_Training_Part3_GroundTruth.csv"
data_histo = np.genfromtxt(path_histo, delimiter=",", dtype=None)
mela = data_histo[1:,1]

mela = list(map(float,mela))
mela = list(map(int, mela))

hist = plt.hist(mela, bins=np.arange(0, 3, 1), color=['b'])
# hist = plt.hist(mela, normed=True, bins='auto')
x = [0.5, 1.5]
plt.xticks(x, ['Benign','Malignant'])
plt.title("ISIC 2017 Melanoma Histogram")
# plt.xlabel("Melanoma")
plt.show()