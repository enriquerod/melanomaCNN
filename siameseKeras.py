'''Train a Siamese MLP on pairs of digits from the MNIST dataset.
It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).
[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
Gets to 99.5% test accuracy after 20 epochs.
3 seconds per epoch on a Titan X GPU
'''
from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop
from keras import backend as K

from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    #it takes the min lenght of all the classes
    # and that number will be the index lenght for each class 
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    print('n> ', n)
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            print(z1, z2)
            input()
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            print(z1, z2)
            input()
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]

    return np.array(pairs), np.array(labels)


def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    return seq


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_dim = 784
epochs = 20
# print('checar1: ', y_train.shape)
# print(y_train[0], y_train[1], y_train[2])
# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(10)]
# dgit_indices has lenght of 10
# each of one has the index for each class
#len(digit_indices) = 10
#digit_indices[0].shape = 5923 which means that it has 5923 index of class 0
#digit_indices[0] =[1 21 34 ... 59952 59972 59987]
#
#
# print('checar: ', len(digit_indices))
# print(digit_indices[0].shape)
# print(digit_indices[0])
tr_pairs, tr_y = create_pairs(x_train, digit_indices)

print('1: ', tr_pairs.shape)
print('2: ', tr_y.shape)

prueba = tr_pairs[3]
print('3: ', prueba.shape)

print('resultado: ',  tr_y[3])
img1 = prueba[0]
img = img1.reshape(28, 28)
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()
img2 = prueba[1]
img = img2.reshape(28, 28)
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()




# digit_indices = [np.where(y_test == i)[0] for i in range(10)]
# te_pairs, te_y = create_pairs(x_test, digit_indices)

# # network definition
# base_network = create_base_network(input_dim)

# input_a = Input(shape=(input_dim,))
# input_b = Input(shape=(input_dim,))

# # because we re-use the same instance `base_network`,
# # the weights of the network
# # will be shared across the two branches
# processed_a = base_network(input_a)
# processed_b = base_network(input_b)

# distance = Lambda(euclidean_distance,
#                   output_shape=eucl_dist_output_shape)([processed_a, processed_b])

# model = Model([input_a, input_b], distance)

# # train
# rms = RMSprop()
# model.compile(loss=contrastive_loss, optimizer=rms)

# graph = plot_model(model, to_file='modelSiamese.png', show_shapes=True)


# model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
#           batch_size=128,
#           epochs=epochs,
#           validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

# # compute final accuracy on training and test sets
# pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
# tr_acc = compute_accuracy(pred, tr_y)
# pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
# te_acc = compute_accuracy(pred, te_y)

# print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
# print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))