from unet import *
from data import *
import matplotlib.pyplot as plt
from keras.models import load_model
# mydata = dataProcess(512,512)

# imgs_test = mydata.load_test_data()

imgs_test = tiff.imread('data/test-volume.tif')
imgs_test = np.array(imgs_test)
plt.imshow(imgs_test[0], cmap='gray')
plt.show()
imgs_test = imgs_test.reshape(imgs_test.shape[0], 512, 512, 1)
imgs_test = imgs_test.astype('float32')
imgs_test /= 255
mean = imgs_test.mean(axis = 0)
imgs_test -= mean	
print(imgs_test.shape)
img = imgs_test[0]
img = img.reshape(1, 512, 512, 1)
print(img.shape)

model = load_model('unet.hdf5', compile=False)
# myunet = myUnet()

# model = myunet.get_unet()

# model.load_weights('unet.hdf5')

# imgs_mask_test = model.predict(img, verbose=1)

# np.save('imgs_mask_test.npy', imgs_mask_test)

print(imgs_mask_test.shape)