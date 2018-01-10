#data interfaces for propofol_cnn
#This is from jsseely/tensorflow-targ-prop
# with additions for the propofol project by pgulley
import pandas as pd
import numpy as np
import random


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

def data_singles(loc):
	chips = []
	labels = []
	for i in pd.DataFrame.from_csv(loc).itertuples():
		
.		chips.append(i[1])
		labels.append(i[2])
	return DataSet(np.asarray(chips), np.asarray(labels))

def train_singles():
	return data_singles("data/train_set.csv")

def test_singles():
	return data_singles("data/test_set.csv")

def data_pairs(data, num_pairs):
	sleep = []
	awake = []
	for entry in data:
		if entry[2] == 1:
			awake.append(entry)
		elif entry[2] == 0:
			sleep.append(entry)

	#If we want repeatable experiments, we'll need to use random.seed()
	random.shuffle(sleep)
	random.shuffle(awake)
	
	chips = []
	labels = []
	
	def add_pair(chip):
		chips.append(chip[0])
		labels.append(chip[1])

	for i in range(num_pairs):
		a = sleep[i]
		b = sleep[-i]
		c = awake[i]
		d = awake[-i]

		
		if i%2 == 0:
			add_pair([[a, b], 1])
			add_pair([[c, d], 1])
			
		if i%2 == 1:
			diff_sets = [[[a, c], 0], [[a, d], 0], [[b, c], 0],[[b, d], 0]]
			random.shuffle(diff_sets)
			add_pair(diff_sets[0])
			add_pair(diff_sets[1])

	return DataSet(np.asarray(chips), np.asarray(labels))



def train_pairs(num):
	data = [i for i in pd.DataFrame.from_csv("data/train_set.csv").itertuples()]
	return data_pairs(data, num)

def test_pairs(num):
	data = [i for i in pd.DataFrame.from_csv("data/test_set.csv").itertuples()]
	return data_pairs(data, num)


def tests():
	"""
	tests for the above. 

	print "ten random train batches size 10"
	for t in range(10):
		A = train_data()
		for i in range(0,10):
			print A.next_batch(10)

	print "ten random test batches size 10"
	for t in range(10):
		A = test_data()
		for i in range(0,10):
			print A.next_batch(10)

	print "let's just see if the randomization works"
	A = train_data()
	last = None
	for t in range(100):
		test = A.next_batch(10)
		if last is not None:
			if np.array_equal(test, last):
				print "same"
			else:
				print "diff"	
		last = test
   
	"""
	print "let's get a few data pairs"
	A = train_pairs(1000)
	for i in range(10):
		print A.next_batch(10)
	
	pass 


if __name__ == "__main__":
	tests()
