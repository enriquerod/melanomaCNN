#no funciona bien, da un nivel de acurrecy muy bajo
#checar como liberar memoria con las variables
#hprueba branch develop
import numpy as np
import os
import glob
import cv2
import math
import matplotlib.pyplot as plt
import time
import re



from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Lambda
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2

from keras.optimizers import sgd
import model as sn

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.layers import Flatten
# from keras.layers.convolutional import Convolution2D
# from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

from keras.utils.vis_utils import plot_model


# fix random seed for reproducibility
# seed = 9
# np.random.seed(seed)
# img = cv2.imread('mnist_training/0.png', 0)
# files = glob.glob('mnist_training/*.png')
# print(files[0])
# img = cv2.imread(files[0], 0)
# print('imprimir segunda')
# plt.imshow(img, cmap = 'Greys', interpolation = 'None')
# plt.show()


def tiny_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)
    print(input_shape)
    # base
    img_input = Input(input_shape)
    print(img_input)
    x = Conv2D(5, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                                            use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(5, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                                            use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    residual = Conv2D(8, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(8, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(8, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 2
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 4
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = Conv2D(num_classes, (3, 3),
            #kernel_regularizer=regularization,
            padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)

    model = Model(img_input, output)
    return model



print('Develop')

img_w = 224
img_h = 224
img_d = 3
def get_im(path):
    # Load as grayscale
    if img_d == 1:
        img = cv2.imread(path,1)
    else:
        img = cv2.imread(path)
    img = cv2.resize(img, (img_w,img_h))
    return img

# def image_to_feature_vector(image, size=(28, 28)):
# 		# resize the image to a fixed size, then flatten the image into
# 		# a list of raw pixel intensities
# 		return cv2.resize(image, size).flatten()

def load_train():
    X_train = []
    X_train_label = []
    print('Read train images and labels')
    files = glob.glob('C:/git/Melanoma_training1/*.jpg')
    for fl in files:
        flbase = os.path.basename(fl)
        flbase = os.path.splitext(flbase)[0]
        img_data = get_im(fl)
        #img=image_to_feature_vector(img,(128,128))
        flbase = re.findall('\d+', flbase)
        # print('label de la imagen',flbase)
        # print('imagen test no. :', fl)
        # plt.imshow(img_data, cmap = 'Greys', interpolation = 'None')
        # plt.show()
        X_train_label.append(flbase)
        X_train.append(img_data)
    #Y_train = np.loadtxt("Training_GroundTruth.csv", delimiter=",", dtype=None)
    Y_train_csv = np.genfromtxt("Training_GroundTruth.csv", delimiter=",", dtype=None)

    # print(Y_train)
    # Y_aqui1 = Y_train[0]
    # print(Y_aqui1)
    # Y_aqui1 = Y_aqui1[1]
    # print(Y_aqui1)

    Y_train = []
    for j in Y_train_csv:
        Y_train.append(j[1])


    #X_train_label = list(map(int, X_train_label))
    return X_train, Y_train, X_train_label

def load_test():
    X_test1 = []
    X_test1_label = []
    print('Read test images and labels')
    files = glob.glob('C:/git/Melanoma_testing1/*.jpg')
    for fl in files:
        flbase = os.path.basename(fl)
        flbase = os.path.splitext(flbase)[0]
        img_data = get_im(fl)
        flbase = re.findall('\d+', flbase)
        # print('label de la imagen',flbase)
        # print('imagen test no. :', fl)
        # plt.imshow(img_data, cmap = 'Greys', interpolation = 'None')
        # plt.show()
        #img=image_to_feature_vector(img,(128,128))
        X_test1_label.append(flbase)
        X_test1.append(img_data)
    Y_test1_csv = np.genfromtxt("Test_GroundTruth.csv", delimiter=",", dtype=None)
    Y_test1 = []
    for j in Y_test1_csv:
        Y_test1.append(j[1])

    #X_test1_label = list(map(int, X_test1_label))
    #X_test1_label.sort()
    return X_test1, Y_test1, X_test1_label


#files = glob.glob('mnist_training/*.png')
#print(files[0])

def normalize_train_data():
    train_data, Y_train_data, train_data_label = load_train()
    train_data = np.array(train_data,  dtype=np.uint8)
    train_data = train_data.reshape(train_data.shape[0], img_w, img_h, img_d)
    train_data = train_data.astype('float32')
    train_data = train_data/255
    #Y_train_data = orderY(train_data_label, Y_train_data)
    Y_train_data = np_utils.to_categorical(Y_train_data)
    num_classes = Y_train_data.shape[1]
    return train_data, Y_train_data, num_classes

def normalize_test_data():
    test_data, Y_test_data , test_data_label= load_test()
    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.reshape(test_data.shape[0], img_w, img_h, img_d)
    test_data = test_data.astype('float32')
    #X_test = preprocessing.scale(X_test)
    test_data = test_data/255

    # = orderY(test_data_label, Y_test_data)

    Y_test_data = np_utils.to_categorical(Y_test_data)
    return test_data, Y_test_data

def orderY(label_try , y_try):
    y_final = []
    for i in label_try:
        y_final.append(y_try[i])
    #y_final = [x for y, x in sorted(zip(y_try, label_try))]
    return y_final


# print(X_test.shape)
# print(Y_test.shape)

# testing_data_bien = np.loadtxt("mnist_train.csv", delimiter=",")
# X_test_bien = testing_data_bien[:,1:]
# X_test_bien = X_test_bien.reshape(X_test_bien.shape[0], 28, 28, 1).astype('float32')
# X_test_bien = X_test_bien/255
# Y_test_bien = testing_data_bien[0:,0]
# Y_test_bien = np_utils.to_categorical(Y_test_bien)
# print('Del dataset bien')
# print(X_test_bien.shape)
# print(Y_test_bien.shape)

# imagen = cv2.imread("mnist_testing/0.png", 0)
# print(imagen)
#print(num_classes)
#print(X_test[0])
#print('Test shape:', X_test.shape)
#print(X_test.shape[0], 'test samples')
#
# #



# print('Del dataset mal')
# print(X.shape)
# print(Y.shape)
# # # plt.imshow(X_test[0], cmap = 'Greys', interpolation = 'None')
# # # plt.show()
# #
# #
# # # print(X_train)
# print('Loading DataSet')
# # load pima indians dataset
# training_data = np.loadtxt("mnist_train.csv", delimiter=",")
# testing_data = np.loadtxt("mnist_test.csv", delimiter=",")
# # split into input (X) and output (Y) variables
# X = training_data[:,1:]
# X_images = X.reshape(X.shape[0], 28, 28).astype('int32')
# X = X.reshape(X.shape[0], 28, 28, 1).astype('float32')
# #plt.imshow(X_images[0], cmap = 'Greys')
# X = X/255
# Y = training_data[0:,0]
# Y = np_utils.to_categorical(Y)
#
#
# X_test = testing_data[:,1:]
# X_tes_images = X_test.reshape(X_test.shape[0], 28, 28).astype('int32')
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
# X_test = X_test/255
# Y_test = testing_data[0:,0]
# Y_test = np_utils.to_categorical(Y_test)
# num_classes = Y_test.shape[1]

#
# X_final, X_test_final, Y_final, Y_test_final = train_test_split(X, Y, test_size=0.2, random_state=2)
X1, Y1, classes = normalize_train_data()
X_test, Y_test = normalize_test_data()

print('No. de clases: ', classes)
print(X1.shape)
print(Y1.shape)

X, X_val, Y, Y_val = train_test_split(X1, Y1, test_size=.30)







# print('Original...')
# Y1 = Y1[0:50,:]

# plt.figure(1)

# # create a grid of 3x3 images
# for i in range(0, 9):
#     plt.subplot(330 + 1 + i)
    
#     if img_d == 1:
#         img_aux = X1[i].reshape(img_w, img_h)
#         plt.imshow(img_aux, cmap=plt.get_cmap('gray'))
#         #plt.imshow(cv2.cvtColor(img_aux, cv2.COLOR_BGR2GRAY))
        
#     else:
#         img_aux = X1[i].reshape(img_w, img_h, img_d)
#         plt.imshow(cv2.cvtColor(img_aux, cv2.COLOR_BGR2RGB))
# # show the plot

# print('Augmenting ...')
# datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# #datagen = ImageDataGenerator(rotation_range=90)
# #datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

# # shift = 0.2
# # datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)

# #datagen = ImageDataGenerator(zca_whitening=True)

# # fit parameters from data
# datagen.fit(X1)
# # configure batch size and retrieve one batch of images
# plt.hold(True)
# plt.figure(2)

# os.makedirs('augmented')
# #, save_prefix='aug'
# for X_batch, y_batch in datagen.flow(X1, Y1, batch_size=50, save_to_dir='augmented', save_format='png'):
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
#     break






def baseline_model():
    # create model
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(img_w, img_h, img_d),
    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
# model = baseline_model()

#Xception
# model = tiny_XCEPTION((img_h, img_w, img_d), 2, l2_regularization=0.01)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Squeezznet

model = sn.SqueezeNet( 2, inputs=(3, 224, 224))
model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])



# Fit the model
print('Training the CNN ...')
graph = plot_model(model, to_file='model.png', show_shapes=True)
model.fit(X, Y, validation_data=(X_val, Y_val), nb_epoch=15, batch_size=100, verbose=2)
#model.fit(X1, Y1, nb_epoch=10, batch_size=30, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))






