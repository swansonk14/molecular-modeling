""" Deep Learning for Molecular Modeling Project """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np 
import json
from generator import Generator
from collections import Counter
np.random.seed(0)
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

""" Load the metadata """
with open('metadata/metadata.json', 'r') as f:
	metadata = json.load(f)

""" Restrict data """
data_size = None
temps = [0.4]
rhos = [0.05, 0.10]

""" Shuffle the data """
np.random.shuffle(metadata)

if temps and rhos:
	metadata = [row for row in metadata if row['label'][0] in temps and row['label'][1] in rhos]
metadata = metadata[:data_size]

for row in metadata:
	row['label'] = tuple(row['label'])

""" Function to convert the labels to one hot """
def get_unique_labels(metadata):
	return list(set([row['label'] for row in metadata]))

def create_one_hot_mapping(unique_labels):
	one_hot_mapping = dict()

	for i, label in enumerate(unique_labels):
		one_hot = np.zeros(len(unique_labels))
		one_hot[i] = 1
		one_hot_mapping[label] = one_hot

	return one_hot_mapping

def convert_to_one_hot(metadata, one_hot_mapping):
	for row in metadata:
		row['original_label'] = row['label']
		row['label'] = one_hot_mapping[row['label']]

	return metadata

""" Convert the data to one hot """
unique_labels = get_unique_labels(metadata)
one_hot_mapping = create_one_hot_mapping(unique_labels)
metadata = convert_to_one_hot(metadata, one_hot_mapping)

""" Define input and output sizes """
im_size = 250
n_outputs = len(unique_labels)

""" Create batch generators for train and test """
train_metadata, test_metadata = train_test_split(metadata, test_size=0.2, random_state=0)

train_generator = Generator(train_metadata, im_size=im_size)
test_generator = Generator(test_metadata, im_size=im_size)

""" Hyperparameters """
batch_size = 128
iterations = 20000
iterations_per_eval = 10
examples_per_eval = 1000
eta = 1e-4
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

def plot(train_accuracies, train_losses, test_accuracies, test_losses):
	plt.subplot(221)
	plt.plot(range(len(train_losses)), train_losses)
	plt.title("Training")
	plt.ylabel('Loss')
	
	plt.subplot(222)
	plt.plot(range(len(test_losses)), test_losses)
	plt.title("Test")
			
	plt.subplot(223)
	plt.plot(range(len(train_accuracies)), train_accuracies)
	plt.ylabel('Accuracy')
	plt.xlabel('Number of iterations')

	plt.subplot(224)
	plt.plot(range(len(test_accuracies)), test_accuracies)
	plt.xlabel('Number of iterations')

	plt.savefig('10x10_5x5_error.png')

def main(_):
	# Input data
	x = tf.placeholder(tf.float32, [None, im_size*im_size])

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
	train_losses = []
	train_accuracies = []
	test_losses = []
	test_accuracies = []

	# Run the network
	with tf.Session(config=config) as sess:

		# Initialize variables
		sess.run(tf.global_variables_initializer())

		# Print class balance
		train_counts = Counter(row['original_label'] for row in train_generator.metadata)
		test_counts = Counter(row['original_label'] for row in test_generator.metadata)

		print('')
		print('class balance')
		print('')
		print('train counts')
		print(train_counts)
		print('test counts')
		print(test_counts)
		print('')

		# Print hyperparameters
		print('iterations = %d, eta = %g, batch_size = %g' % (iterations, eta, batch_size))
		print('temperatures')
		print(temps)
		print('densities')
		print(rhos)

		# Training
		print('Training')
		for i in range(iterations):
			if i % 5 == 0:
				print('iteration {}'.format(i))

			# Evaluate
			if i % 25 == 0:
				print('')
				print('Evaluating')
				# Evaluate on train set
				train_batch_accuracies = []
				train_batch_losses = []
				for train_X, train_Y in train_generator.data_in_batches(examples_per_eval, batch_size):
					train_batch_accuracies.append(accuracy.eval(feed_dict={
							x: train_X, y_: train_Y, keep_prob: 1.0}))

					train_batch_losses.append(loss.eval(feed_dict={
							x: train_X, y_: train_Y, keep_prob: 1.0}))

				train_accuracy = np.mean(train_batch_accuracies)
				train_loss = np.mean(train_batch_losses)

				train_accuracies.append(train_accuracy)
				train_losses.append(train_loss)

				# Evaluate on test set
				test_batch_accuracies = []
				test_batch_losses = []
				for test_X, test_Y in test_generator.data_in_batches(examples_per_eval, batch_size):
					test_batch_accuracies.append(accuracy.eval(feed_dict={
							x: test_X, y_: test_Y, keep_prob: 1.0}))

					test_batch_losses.append(loss.eval(feed_dict={
							x: test_X, y_: test_Y, keep_prob: 1.0}))

				test_accuracy = np.mean(test_batch_accuracies)
				test_loss = np.mean(test_batch_losses)

				test_accuracies.append(test_accuracy)
				test_losses.append(test_loss)

				print('step %d, training accuracy %g, train loss %g, ' \
					'test accuracy %g, validation loss %g' %
					(i, train_accuracy, train_loss, test_accuracy, test_loss))
				print('')

				plot(train_accuracies, train_losses, test_accuracies, test_losses)
			
			# Train
			train_X, train_Y = train_generator.next(batch_size)
			train_step.run(feed_dict={x: train_X, y_: train_Y, keep_prob: 0.5})

	plot(train_accuracies, train_losses, test_accuracies, test_losses)

# Run the program 
if __name__ == '__main__':
	tf.app.run(main=main)
