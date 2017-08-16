from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import system things
from tensorflow.examples.tutorials.mnist import input_data # for data
import tensorflow as tf
import numpy as np
import os


def conv_fc_layer(img, kernel_shape):
    #create weights ('with names')
    #prepare params densely connected layer

    weights_for_convolution = tf.get_variable("weights_for_convolution", kernel_shape,
        initializer=tf.random_normal_initializer())

    weights_for_connected_layer = tf.get_variable("weights_for_connected_layer", [14*14*32,1024],
        initializer=tf.random_normal_initializer())

    #weights_for_readout_layer = tf.get_variable("weights_for_readout_layer", [1024,2],
        #initializer=tf.random_normal_initializer())

    weights_for_readout_layer = tf.get_variable("weights_for_readout_layer", [1024,2],
        initializer=tf.random_normal_initializer())

    #create biases ('with names')

    bias_shape = kernel_shape[-1]
    biases_for_convolution = tf.get_variable("biases_for_convolution", [bias_shape],
        initializer=tf.constant_initializer(0.1))


    biases_for_connected_layer = tf.get_variable("biases_for_connected_layer", [1024],
        initializer=tf.constant_initializer(0.1))

    #biases_for_readout_layer = tf.get_variable("biases_for_readout_layer", [2],
        #initializer=tf.constant_initializer(0.1))


    biases_for_readout_layer = tf.get_variable("biases_for_readout_layer", [2],
        initializer=tf.constant_initializer(0.1))

    #rechaping Image
    x_image = tf.reshape(img,[-1,28,28,1])

    #convolution and poooling

    c2 = tf.nn.conv2d(x_image, weights_for_convolution, strides=[1, 1, 1, 1], padding='SAME')
    conv = tf.nn.bias_add(c2, biases_for_convolution)

    #print("hello")

    relu = tf.nn.relu(conv)
    out = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME')


    #densely connected layer
        #2. flatten the previous output

    h_out_flat = tf.reshape(out ,[-1,14*14*32])

    #multiply by a weight matrix, add a bias, and apply a ReLU.

    h_fc1 = tf.nn.relu(tf.matmul(h_out_flat, weights_for_connected_layer) + biases_for_connected_layer)

    #compute model output

    final_output = tf.matmul(h_fc1,weights_for_readout_layer) + biases_for_readout_layer


    #pass

    return final_output


def make_model(x1 , x2):
    with tf.variable_scope("network") as scope: #here to check if there is an error #with tf.variable_scope("network"):gives diffrent resul
        network1 = conv_fc_layer(x1, [5, 5, 1, 32])
        #print('hello1')
        # Variables created here should be named:
                                                #1. "network1/weights", "conv1/biases"

        scope.reuse_variables()
        #print("hello")
        network2 = conv_fc_layer(x2, [5, 5, 1, 32])

    return network1 , network2

def step_loss(y_ , network1_output , network2_output):
        margin = 5.0
        labels_t = y_
        labels_f = tf.subtract(1.0, y_, name="1-yi")  #choose what is convenient to your classes # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(network1_output, network2_output), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.multiply(labels_t, eucd, name="y_x_eucd")
        neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C, eucd)), name="Ny_C-eucd")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss


    #pass
        #print("hello2")
        # Variables created here should be named
                                                 #2. "conv2/weights", "conv2/biases".
    # ...
    #assert conv1 is conv2

    #tf.reset_default_graph()

    #tf.reset_default_graph()
tf.reset_default_graph()
#x = tf.placeholder(tf.float32, [None,784])
x1 = tf.placeholder(tf.float32, [None, 784])
x2 = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32,[None])

network1 , network2 = make_model(x1 , x2)

#for v in tf.trainable_variables():
    #print('hello trainable variables')
    #print( v.name, v.get_shape().as_list())
#tf.reset_default_graph()



# prepare data and tf.session
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
#sess = tf.InteractiveSession()

network_loss = step_loss(y_ , network1 , network2)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(network_loss)
saver = tf.train.Saver()


with tf.Session() as sess:
    tf.global_variables_initializer().run()
# start training
    for step in range(500):
        batch_x1, batch_y1 = mnist.train.next_batch(128)
        batch_x2, batch_y2 = mnist.train.next_batch(128)
        batch_y = (batch_y1 == batch_y2).astype('float')
        #batch_y = batch_y1

        _, loss_v = sess.run([train_step, network_loss], feed_dict={
                                   x1: batch_x1,
                                   x2: batch_x2,
                                   y_: batch_y})


        if step % 10 == 0:
            print ('step %d: loss %.3f' % (step, loss_v))
