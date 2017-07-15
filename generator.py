from scipy.misc import imread, imresize
import numpy as np

class Generator():
	def __init__(self, metadata, im_size):
		self.metadata = metadata
		np.random.shuffle(self.metadata)
		self.im_size = im_size
		self.im_shape = (im_size, im_size)
		self.index = 0

	def next(self, batch_size=None):
		if batch_size is None:
			batch_size = len(self.metadata)

		# get the next batch_size rows from data
		if self.index + batch_size <= len(self.metadata):
			batch = self.metadata[self.index:self.index + batch_size]
			self.index += batch_size
		else:
			batch = self.metadata[self.index:]
			diff = batch_size - (len(self.metadata) - self.index)
			np.random.shuffle(self.metadata)
			batch += self.metadata[:diff]
			self.index = diff

		# get the images and labels for the batch
		images = [None]*batch_size
		labels = [None]*batch_size
		for i in range(len(batch)):
			row = batch[i]
			images[i] = imresize(imread(row['path']), self.im_shape).flatten()
			labels[i] = row['label']

		images = np.array(images)
		labels = np.array(labels)

		return images, labels

	def data_in_batches(self, num_examples=None, batch_size=None):
		if num_examples is None:
			num_examples = len(self.metadata)
		num_examples = min(num_examples, len(self.metadata))

		if batch_size is None:
			batch_size = len(self.metadata)

		self.index = 0
		np.random.shuffle(self.metadata)

		while self.index + batch_size <= num_examples:
			yield self.next(batch_size)
