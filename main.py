""" Deep Learning for Molecular Modeling Project 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tensorflow as tf
import numpy as np 
import math
import os
import time
import random
import scipy
from scipy.misc import imread
import json
from KirkSwanson_batch_Scratch import Batch
np.random.seed(0)
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

""" Load the metadata """
data = json.load(open('metadata/metadata.json', 'r'))

""" Restrict data """
data_size = None
temps = [0.4]
rhos = [0.05, 0.10]

if temps and rhos:
  data = list(filter(lambda x: x['label'][0] in temps and x['label'][1] in rhos, data))
if data_size:
  data = data[:data_size]

""" Shuffle the data """
np.random.shuffle(data)

""" Function to convert the data to one hot """
def create_one_hot_mapping(dataset):
  unique_labels = list(set([tuple(row['label']) for row in dataset]))
  one_hot_mapping = dict()

  for i in range(len(unique_labels)):
    label = unique_labels[i]
    one_hot = np.zeros(len(unique_labels))
    one_hot[i] = 1
    one_hot_mapping[label] = one_hot

  return one_hot_mapping

def convert_to_one_hot(dataset):
  one_hot_mapping = create_one_hot_mapping(dataset)
  for row in dataset:
    row['label'] = one_hot_mapping[tuple(row['label'])]
  return dataset

""" Convert the data to one hot """
data = convert_to_one_hot(data)

""" Create batch generators for train and test """
train = Batch(data[:int(0.8*len(data))])
validation = Batch(data[int(0.8*len(data)):int(0.95*len(data))])
test = Batch(data[int(0.95*len(data)):])

""" Misc. variables """
n_outputs = train.label_size

FLAGS = None

""" Hyperparameters """
iterations = 20000
eta = 1e-4
batch_size = 30
beta = 0.01

""" Function that builds the graph for the neural network """
def deepnn(x):
  # First convolutional layer
  x_image = tf.reshape(x, [-1, 250, 250, 1])
  W_conv1 = weight_variable([10, 10, 1, 6])
  b_conv1 = bias_variable([6])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
 
  # Second convolutional layer
  W_conv2 = weight_variable([5, 5, 6, 16])
  b_conv2 = bias_variable([16])
  h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

  # Fully connected layer
  W_fc1 = weight_variable([237 * 237 * 16, 80])
  b_fc1 = bias_variable([80])
  h_conv2_flat = tf.reshape(h_conv2, [-1, 237*237*16])
  h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

  # Dropout on the fully connected layer
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, seed=0)

  # Output layer
  W_fc2 = weight_variable([80, n_outputs])
  b_fc2 = bias_variable([n_outputs])
  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  # Returns the prediction and the dropout probability placeholder
  return y_conv, keep_prob, W_conv1, W_conv2, W_fc1, W_fc2


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1, seed=0)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  # Input data
  x = tf.placeholder(tf.float32, [None, train.im_size])

  # Output
  y_ = tf.placeholder(tf.float32, [None, n_outputs])

  # Build the graph for the deep net
  y_conv, keep_prob, W_conv1, W_conv2, W_fc1, W_fc2 = deepnn(x)

  # Define the los and the optimizer
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  regularizers = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)
  loss = tf.reduce_mean(loss + beta*regularizers)
  train_step = tf.train.AdamOptimizer(eta).minimize(loss)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Save GPU memory preferences
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  """ Lists for plotting """
  train_iterations = []
  train_errors = []
  train_accuracies = []
  validation_iterations = []
  validation_errors = []
  validation_accuracies = []

  # Run the network
  with tf.Session(config=config) as sess:

    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Print hyperparameters
    print('iterations = %d, eta = %g, batch_size = %g' % (iterations, eta, batch_size))
    print('temperatures')
    print(temps)
    print('densities')
    print(rhos)

    # Training
    for i in range(iterations):
      train_X, train_Y = train.next(batch_size)
      if i % 100 == 0 and i > 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: train_X, y_: train_Y, keep_prob: 1.0})
        train_iterations.append(i)
        train_accuracies.append(train_accuracy)
        train_errors.append(loss.eval(feed_dict={
            x: train_X, y_: train_Y, keep_prob: 1.0}))
        validation_accuracy = []
        validation_loss = []
        for j in range(int(len(validation.dataset)/batch_size)):
          validation_X, validation_Y = validation.next(batch_size)
          validation_accuracy.append(accuracy.eval(feed_dict={
            x: validation_X, y_: validation_Y, keep_prob: 1.0}))
          validation_loss.append(loss.eval(feed_dict={
            x: validation_X, y_: validation_Y, keep_prob: 1.0}))
        validation_iterations.append(i)
        validation_accuracies.append(float(sum(validation_accuracy))/float(len(validation_accuracy)))
        validation_errors.append(float(sum(validation_loss))/float(len(validation_loss)))
        print('step %d, training accuracy %g, validation accuracy %g, validation loss %g' % (i, train_accuracy, float(sum(validation_accuracy))/float(len(validation_accuracy)), float(sum(validation_loss))/float(len(validation_loss))))
      train_step.run(feed_dict={x: train_X, y_: train_Y, keep_prob: 0.5})

    # Testing
    test_accuracy = []
    for j in range(int(len(test.dataset)/batch_size)):
      test_X, test_Y = test.next(batch_size)
      test_accuracy.append(accuracy.eval(feed_dict={
        x: test_X, y_: test_Y, keep_prob: 1.0}))
    print('test accuracy %g' % (float(sum(test_accuracy))/float(len(test_accuracy))))

    """ Plot results """
    plt.subplot(221)
    plt.plot(train_iterations, train_errors)
    plt.title("Training")
    plt.ylabel('Loss')
    
    plt.subplot(222)
    plt.plot(validation_iterations, validation_errors)
    plt.title("Validation")
        
    plt.subplot(223)
    plt.plot(train_iterations, train_accuracies)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of iterations')

    plt.subplot(224)
    plt.plot(validation_iterations, validation_accuracies)
    plt.xlabel('Number of iterations')

    plt.savefig('10x10.5x5.error.png')

# Run the program 
if __name__ == '__main__':
  tf.app.run(main=main)
