import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
# from data import dataProcess

from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard

# from PIL import Image
# import tifffile as tiff
import scipy.io
from scipy.io import loadmat

class myUnet(object):

	def load_train_data(self):
		# im = Image.open('data/train-volume.tif')
		imgs_train = tiff.imread('data/train-volume.tif')
		imgs_train = np.array(imgs_train)
		imgs_mask_train = tiff.imread('data/train-labels.tif')
		imgs_mask_train = np.array(imgs_mask_train)

		# print('-'*30)
		# print('load train images...')
		# print('-'*30)
		# imgs_train = np.load(self.npy_path+"/imgs_train.npy")
		# imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train.npy")
		imgs_train = imgs_train.reshape(imgs_train.shape[0], 512, 512, 1)
		imgs_mask_train = imgs_mask_train.reshape(imgs_mask_train.shape[0], 512, 512, 1)
		
		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
		imgs_train /= 255
		mean = imgs_train.mean(axis = 0)
		imgs_train -= mean	
		imgs_mask_train /= 255
		imgs_mask_train[imgs_mask_train > 0.5] = 1
		imgs_mask_train[imgs_mask_train <= 0.5] = 0

		return imgs_train,imgs_mask_train

	# def __init__(self, img_rows = 512, img_cols = 512):
	# def __init__(self, img_rows = 128, img_cols = 128):
	def __init__(self, img_rows = 256, img_cols = 256):

		self.img_rows = img_rows
		self.img_cols = img_cols

	# def load_data(self):

	# 	mydata = dataProcess(self.img_rows, self.img_cols)
	# 	imgs_train, imgs_mask_train = mydata.load_train_data()
	# 	imgs_test = mydata.load_test_data()
	# 	return imgs_train, imgs_mask_train, imgs_test

	def get_unet(self):

		inputs = Input((self.img_rows, self.img_cols,1))
		# inputs = Input((self.img_rows, self.img_cols,3))
				
		
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		print ("conv1 shape:",conv1.shape)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		print ("conv1 shape:",conv1.shape)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		print ("pool1 shape:",pool1.shape)

		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		print ("conv2 shape:",conv2.shape)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		print ("conv2 shape:",conv2.shape)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		print ("pool2 shape:",pool2.shape)

		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		print ("conv3 shape:",conv3.shape)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		print ("conv3 shape:",conv3.shape)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		print ("pool3 shape:",pool3.shape)

		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		drop5 = Dropout(0.5)(conv5)

		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

		model = Model(input = inputs, output = conv10)

		model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

		return model


	def train(self):
		print("LOADING DATA...")
		# data = loadmat("dataset.mat")
		data_t = loadmat("data_train_256.mat")
		# data = loadmat("100.mat")
		imgs_train = data_t["train"]
		imgs_mask_train = data_t["mask"]

		data_v = loadmat("data_val_256.mat")
		imgs_val = data_v["train"]
		imgs_mask_val = data_v["mask"]

		# self.load_train_data()
		# print("loñading data")
		# imgs_train, imgs_mask_train, imgs_test = self.load_data()
		# imgs_train, imgs_mask_train = self.load_train_data()
		print("Done!!")
		print("Normalizing DATA...")
		# imgs_train = imgs_train.reshape(imgs_train.shape[0], 128, 128, 1)
		# imgs_mask_train = imgs_mask_train.reshape(imgs_mask_train.shape[0], 128, 128, 1)

		imgs_train = imgs_train.reshape(imgs_train.shape[0], 256, 256, 1)
		imgs_mask_train = imgs_mask_train.reshape(imgs_mask_train.shape[0], 256, 256, 1)
		
		imgs_val = imgs_val.reshape(imgs_val.shape[0], 256, 256, 1)
		imgs_mask_val = imgs_mask_val.reshape(imgs_mask_val.shape[0], 256, 256, 1)


		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
		imgs_train /= 255
		# mean = imgs_train.mean(axis = 0)
		# imgs_train -= mean	
		imgs_mask_train /= 255
		imgs_mask_train[imgs_mask_train > 0.5] = 1
		imgs_mask_train[imgs_mask_train <= 0.5] = 0


		imgs_val = imgs_val.astype('float32')
		imgs_mask_val = imgs_mask_val.astype('float32')
		imgs_val /= 255
		# mean = imgs_train.mean(axis = 0)
		# imgs_train -= mean	
		imgs_mask_val /= 255
		imgs_mask_val[imgs_mask_val > 0.5] = 1
		imgs_mask_val[imgs_mask_val <= 0.5] = 0
		print("DONE!!!")

		model = self.get_unet()
		print("GOT unet-model, 256 Gray with val")

		print("Dataset Length: ", imgs_train.shape)
		filepath="weights-{epoch:02d}.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='acc', save_best_only=True, save_weights_only=True, mode='max')

		csv_logger = CSVLogger('history.csv')
		# histogram = TensorBoard(histogram_freq=0)
		# callbacks_list = [checkpoint, csv_logger, histogram]

		callbacks_list = [checkpoint, csv_logger]
		# model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
		# print('Fitting model...')
		# model.fit(imgs_train, imgs_mask_train, batch_size=1, nb_epoch=150, verbose=1, shuffle=True, callbacks=[model_checkpoint, csv_logger])
		print("Training model...")
		model.fit(imgs_train, imgs_mask_train, validation_data=(imgs_val, imgs_mask_val), batch_size=1, nb_epoch=50, verbose=1, shuffle=True, callbacks=callbacks_list)
		print("Trianing FINISHED!!!!")
		# print('predict test data')
		# imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
		# np.save('imgs_mask_test.npy', imgs_mask_test)


if __name__ == '__main__':
	myunet = myUnet()
	myunet.train()








