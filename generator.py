from scipy.misc import imread
import numpy as np

class Batch():
	def __init__(self, dataset):
		self.dataset = dataset
		self.index = 0
		self.im_size = len(imread(dataset[0]['path']).flatten())
		self.label_size = len(dataset[0]['label'])

	def next(self, batch_size=None):
		if batch_size is None:
			batch_size = len(self.dataset)

		# get the next batch_size rows from data
		if self.index + batch_size <= len(self.dataset):
			batch = self.dataset[self.index:self.index + batch_size]
			self.index += batch_size
		else:
			diff = batch_size - (len(self.dataset) - self.index)
			batch = self.dataset[self.index:] + self.dataset[:diff]
			self.index = diff

		# get the images and labels for the batch
		images = np.zeros((batch_size, self.im_size))
		labels = np.zeros((batch_size, self.label_size))
		for i in range(len(batch)):
			row = batch[i]
			images[i, :] = imread(row['path']).flatten()
			labels[i, :] = row['label']

		return images, labels