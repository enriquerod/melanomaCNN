
from __future__ import print_function
from random import shuffle
import glob
shuffle_data = True  # shuffle the addresses before saving


import numpy as np
import os
import glob
import cv2
import math
import matplotlib.pyplot as plt
import time
import re
import sys
import tensorflow as tf


from sklearn.cross_validation import train_test_split
from sklearn import preprocessing



img_w = 28
img_h = 28
img_d = 1

# image_filenames = glob.glob("C:/git/melanoma_training/*.jpg")
# print(image_filenames[0:2])


data_path = 'train.tfrecords'  # address to save the hdf5 file
with tf.Session() as sess:
	feature = {'train/image': tf.FixedLenFeature([], tf.string),
	           'train/label': tf.FixedLenFeature([], tf.int64), 
	           'train/name': tf.FixedLenFeature([], tf.string)}
	# Create a list of filenames and pass it to a queue
	filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
	# Define a reader and read the next record
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	# Decode the record read by the reader
	features = tf.parse_single_example(serialized_example, features=feature)
	# Convert the image data from string back to the numbers
	image = tf.decode_raw(features['train/image'], tf.float32)
	# Cast label data into int32
	label = tf.cast(features['train/label'], tf.int32)
	name = features['train/name']
	# Reshape image data into the original shape
	image = tf.reshape(image, [img_w, img_h, img_d])
#
#
#
#
	# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	# sess.run(init_op)
	# # Create a coordinator and run all QueueRunner objects
	# coord = tf.train.Coordinator()
	# threads = tf.train.start_queue_runners(coord=coord)
	# img, lbl, nam = sess.run([image, label, name])

	# print(img.shape)
	# print(lbl)
	# print(nam)
	# img = img.reshape(img_w, img_h)
	# print(img.shape)
	# plt.imshow(img, cmap=plt.get_cmap('gray'))
	# plt.show()
#
#
#
#
	# coord.request_stop()
	# # Wait for threads to stop
	# coord.join(threads)
	# sess.close()


	# Any preprocessing here ...
	# Creates batches by randomly shuffling tensors
	images, labels, names = tf.train.shuffle_batch([image, label, name], batch_size=1, capacity=40, num_threads=1, min_after_dequeue=1)
	# Initialize all global and local variables
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	sess.run(init_op)
	# Create a coordinator and run all QueueRunner objects
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	for batch_index in range(39):
	    img, lbl, nam = sess.run([images, labels, names])
	    print(batch_index)
	    print(lbl)
	    print(nam)
	    # img = img.astype(np.uint8)
	    # for j in range(6):
	    #     plt.subplot(2, 3, j+1)
	    #     plt.imshow(img[j])
	    #     plt.title('cat' if lbl[j]==0 else 'dog')
	    # plt.show()
	# Stop the threads

	coord.request_stop()

	# Wait for threads to stop
	coord.join(threads)
	sess.close()


