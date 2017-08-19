
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



data_path = 'tfrecords'  # address to save the hdf5 file
with tf.Session() as sess:
	feature = {'train/image': tf.FixedLenFeature([], tf.string),
	           'train/label': tf.FixedLenFeature([], tf.int64)}
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
	# Reshape image data into the original shape
	image = tf.reshape(image, [img_w, img_h, img_d])


	# label.shape
	# print('hola', label.shape)
	# print(label[0])
	# #sess.run(label)
	# print('si')
	# sess.run(tf.equal(image, tf_record_image))
	# sess.run(label)
	# sess.close()

    # Any preprocessing here ...
    
	#Creates batches by randomly shuffling tensors

	# images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)
	# # images, labels = tf.train.shuffle_batch([image, label])
	# Initialize all global and local variables
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	sess.run(init_op)
	# Create a coordinator and run all QueueRunner objects
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	print('BIEN')
	label_final = sess.run(label)
	print('holq como estas')
	# for batch_index in range(5):
	#     img, lbl = sess.run([images, labels])
	#     img = img.astype(np.uint8)
	#     for j in range(6):
	#         plt.subplot(2, 3, j+1)
	#         plt.imshow(img[j, ...])
	#         plt.title('cat' if lbl[j]==0 else 'dog')
	#     plt.show()
	# Stop the threads
	coord.request_stop()

	# Wait for threads to stop
	coord.join(threads)
	sess.close()


# # print(X1[20])
# # print(Y1[20])
# # print(L1[20])