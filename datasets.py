#data interfaces for propofol_cnn
#This is from jsseely/tensorflow-targ-prop
# with additions for the propofol project by pgulley
import pandas as pd
import numpy as np
import random
import os


class DataSet(object):
  def __init__(self,
               inputs,
               outputs=None):
    """
      Construct a DataSet object.
      Adapted from mnist.py from the TensorFlow code base.
      Inputs: a shape (N, d_1) numpy array; N samples of a d_1-dimensional vector.
      Outputs: a shape (N, d_2) numpy array; N samples of a d_2-dimensional vector.
    """
    self._inputs = inputs
    self._outputs = outputs
    self._num_examples = inputs.shape[0]
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def inputs(self):
    return self._inputs
  
  @property
  def outputs(self):
    return self._outputs

  @inputs.setter
  def inputs(self, value):
    self._inputs = value

  @outputs.setter
  def outputs(self, value):
    self._outputs = value

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed
  
  def rand_batch(self, batch_size):
    """
      Return a random batch of input / outputs.
      x_batch, y_batch = DataSet.rand_batch(batch_size) if outputs is not None
      x_batch = DataSet.rand_batch(batch_size) if outputs is None
    """
    inds = np.random.choice(self._num_examples, batch_size)
    if self._outputs is None:
      return self._inputs[inds]
    else:
      return self._inputs[inds], self._outputs[inds]

  def next_batch(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      np.random.seed(self._epochs_completed) # ensure the same shuffling behavior for each experiment.
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._inputs = self._inputs[perm]
      self._outputs = self._outputs[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    if self._outputs is None:
      return self._inputs[start:end]
    else:
      return self._inputs[start:end], self._outputs[start:end]



def test_singles():
	#awake
	for root, dirs, filenames in os.walk("data/test/awake"):
		awake = [(np.load(os.path.join(root, file_)), [1,0]) for file_ in filenames]
	for root, dirs, filenames in os.walk("data/test/asleep"):
		asleep = [(np.load(os.path.join(root, file_)), [0,1]) for file_ in filenames]
	
	random.shuffle(awake)
	merged = [j for i in zip(awake,asleep) for j in i]
	chips, labels = zip(*merged) #zip(*) is inverse of zip()- so [(a,1), (b,2), (c,3)] -> [(a, b, c), (1, 2, 3)]
	return DataSet(np.asarray(chips), np.asarray(labels, dtype=np.float64))

""" #commented out because there is no train data right now
def train_singles():
	for root, dirs, filenames in os.walk("data/train/awake"):
		awake = [(np.load(os.join(root, file_)), 1) for file_ in filenames]
	for root, dirs, filenames in os.walk("data/train/asleep"):
		asleep = [(np.load(os.join(root, file_)), 0) for file_ in filenames]
	random.shuffle(awake)
	merged = [j for i in zip(awake,asleep) for j in i]
	chips, labels = zip(*merged) #zip(*) is inverse of zip()- so [(a,1), (b,2), (c,3)] -> [(a, b, c), (1, 2, 3)]
	return DataSet(np.asarray(chips), np.asarray(labels))
"""


def data_pairs(awake, asleep, num_pairs):
	#If we want repeatable experiments, we'll need to use random.seed()
	random.shuffle(asleep)
	random.shuffle(awake)
	
	chips = []
	labels = []
	
	def add_pair(chip):
		chips.append(chip[0])
		labels.append(chip[1])

	for i in range(num_pairs):
		#basically we generate the pairs by popping both the first and last elements 
		#it limits the number of unique pairs we can generate
		#but we it's a limit on how many pairs we can create w/o repeating any singles.
		a = asleep[i]
		b = asleep[-i]
		c = awake[i]
		d = awake[-i]

		#we choose whether to make a "same" pair or a "different" pair
		#with mod2 of the iteration. We add two pairs each time. final data is split 50/50 between same and diff. 
		if i%2 == 0:
			add_pair([[a, b], 1])
			add_pair([[c, d], 1])
			
		if i%2 == 1:
			diff_sets = [[[a, c], 0], [[a, d], 0], [[b, c], 0],[[b, d], 0]]
			random.shuffle(diff_sets)
			add_pair(diff_sets[0])
			add_pair(diff_sets[1])

	return DataSet(np.asarray(chips), np.asarray(labels))


""" #Again, these don't exist rn.
def train_pairs(num):
	for root, dirs, filenames in os.walk("data/train/awake"):
		awake = [(np.load(os.join(root, file_)), 1) for file_ in filenames]
	for root, dirs, filenames in os.walk("data/train/asleep"):
		asleep = [(np.load(os.join(root, file_)), 0) for file_ in filenames]

	return data_pairs(awake, asleep, num)
"""

def test_pairs(num):
	for root, dirs, filenames in os.walk("data/test/awake"):
		awake = [(np.load(os.path.join(root, file_)), [1., 0.]) for file_ in filenames]
	for root, dirs, filenames in os.walk("data/test/asleep"):
		asleep = [(np.load(os.path.join(root, file_)), [0., 1.]) for file_ in filenames]

	return data_pairs(awake, asleep, num)


def tests():

	#tests for the above functions. 

	print "ten random singlet batches"
	for t in range(10):
		A = test_singles()
		for i in range(0,10):
			print A.next_batch(10)

   
	
	print "ten random paired batches"
	A = test_pairs(10)
	for i in range(10):
		print A.next_batch(10)


if __name__ == "__main__":
	tests()
