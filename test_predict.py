from unet import *
# from data import *
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2

# data = loadmat("dataset.mat")
# imgs_train = data["train"]
# img = imgs_train[0]
# plt.imshow(img, cmap='gray')
# plt.show()


# img = cv2.imread("C:/Users/EHO085/Desktop/data_skin/ISIC-2017_Validation_Data/ISIC_0012684.jpg")
# img = cv2.imread("C:/Users/EHO085/Desktop/data_skin/ISIC-2017_Validation_Data/ISIC_0001871.jpg")
img = cv2.imread("C:/Users/EHO085/Desktop/data_skin/ISIC-2017_Validation_Data/ISIC_0009995.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (256,256))
# imgp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img, cmap='gray')
plt.show()


# mydata = dataProcess(512,512)

# imgs_test = mydata.load_test_data()

# imgs_test = tiff.imread('data/test-volume.tif')
# # imgs_test = tiff.imread('data/train-volume.tif')
# imgs_test = np.array(imgs_test)
# print("AQUI",imgs_test.shape)
# print(imgs_test[0])
# plt.imshow(imgs_test[2], cmap='gray')
# plt.show()
# imgs_test = imgs_test.reshape(imgs_test.shape[0], 512, 512, 1)
# imgs_test = imgs_test.astype('float32')
# imgs_test /= 255
# mean = imgs_test.mean(axis = 0)
# imgs_test -= mean	
# print(imgs_test.shape)
# img = imgs_test[2]
img = img.reshape(1, 256, 256, 1)
print(img.shape)

# model = load_model('unet.hdf5', compile=False)
myunet = myUnet()

model = myunet.get_unet()

# model.load_weights('weights-00.hdf5')
# model.load_weights('weights-18.hdf5')
# model.load_weights('weights-val-gray-256-aug/weights-09.hdf5')
model.load_weights('weights-val-gray-256-aug/weights-11.hdf5')

imgs_mask_test = model.predict(img, verbose=1)
print(imgs_mask_test[0])
imgs_mask_test[imgs_mask_test > 0.5] = 255
imgs_mask_test[imgs_mask_test <= 0.5] = 0

# np.save('imgs_mask_test.npy', imgs_mask_test)

print(imgs_mask_test.shape)

img_pred = imgs_mask_test[0]
img_pred = img_pred.reshape(256, 256)
# img_pred = img_pred*255
img_pred = img_pred.astype(np.uint8)
print(img_pred)
print("AQUI2",img_pred.shape)
plt.imshow(img_pred, cmap='gray')
plt.show()