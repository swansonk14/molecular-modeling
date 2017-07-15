from scipy.misc import imread, imresize
import numpy as np

class Generator():
	def __init__(self, metadata, im_size):
		self.metadata = metadata
		self.im_size = im_size
		self.index = 0

	def next(self, batch_size=None):
		if batch_size is None:
			batch_size = len(self.metadata)

		# get the next batch_size rows from data
		if self.index + batch_size <= len(self.metadata):
			batch = self.metadata[self.index:self.index + batch_size]
			self.index += batch_size
		else:
			diff = batch_size - (len(self.metadata) - self.index)
			batch = self.metadata[self.index:] + self.metadata[:diff]
			self.index = diff

		# get the images and labels for the batch
		images = [None]*batch_size
		labels = [None]*batch_size
		for i in range(len(batch)):
			row = batch[i]
			images[i] = imresize(imread(row['path']), (im_size, im_size)).flatten()
			labels[i] = row['label']

		images = np.array(images)
		labels = np.array(labels)

		return images, labels

	def all_data(self, batch_size=None):
		if batch_size is None:
			batch_size = len(self.metadata)

		index = self.index
		self.index = 0

		while self.index + batch_size <= len(self.metadata):
			yield self.next(batch_size)

		self.index = index
