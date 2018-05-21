# Copyright (c) 2017 Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

#Substantial portions of this file are copied freom Ira Ktena's project, with small adjustments made for our workflow. 

import tensorflow as tf
import scipy
import numpy as np
import graph 
import datasets
import re

class deep_cgnn(object):

    def __init__(self, graph, L, inputs, batch_size):
        self.graph = graph
            
        self.inputs = inputs #this should be a DataSet Object
        self.data = [inputs.next_batch(1) for i in range(inputs.num_examples)]
        self.L = L
        self.batch_size = batch_size
        #import pdb; pdb.set_trace()

        self.in_size = [batch_size, 234,234 ]
        self.regularizers = []

        self.F = [64,64] #Features per gcnn layer
        self.p = [1,1] #pooling dim per gcnn layer
        self.K = [3,3] #K approx iters 
        self.M = [1] #output dim
    
       
        self.ph_data = tf.placeholder(tf.float64, self.in_size, 'data')
        self.ph_labels = tf.placeholder(tf.int64, [1], 'labels')
        self.ph_site = tf.placeholder(tf.float64, [1], 'site')
        self.ph_dropout = tf.placeholder(tf.float64, [], 'do_rate')

        self.model = self.build_model(self.ph_data)

    def chebyshev5(self, x, L, Fout, K):
        Fin, N, M = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        #import pdb; pdb.set_trace()
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        #copies into new L (ugh var names)
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat(0, [x, x_])  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        out = tf.reshape(x, [Fout, N, M])
        return out  # N x M x Fout
        """
        Notes on the above function: impliments a wavelet graph transform (as described in hammond 2011)- 
            it's sort of the stand-in for convolution in the graph-signal case: x is transformed into l-space and is 
            54-60 are the recursive chebychev approximation- T_k(C) = 2*T_k-1(C) = T_k-2(c). 
            The paper describes the filtering function as SUM((0,k), W_k*T_k(L)*x) - where W_k is the k localized portion of the filter. 
                The code above seems to have multiplied out the filtering to the last step- it actually happens on line 66. 
            So it's more like SUM((0,k), T-k(L)*x)*F - but the idea is the same. I suppose the details of scaling the filters
            are probably managed by backprop somehow. Sometimes things are magic. 
        """



    def _weight_variable(self, shape, regularization=True):
        """Glorot & Bengio (AISTATS 2010) init."""
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform_initializer(minval=-init_range, maxval=init_range)

        var = tf.get_variable('weights', shape, tf.float64, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
            tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        # initial = tf.constant_initializer(0.1)
        initial = tf.constant_initializer(0.0)

        var = tf.get_variable('bias', shape, tf.float64, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
            tf.summary.histogram(var.op.name, var)
        return var

    def _conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def b1relu(self, x, relu=True):
        """Bias and ReLU (if relu=True). One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        x = x + b
        return tf.nn.relu(x) if relu else x

    def mpool1(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            #tf.maximum
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def build_model(self, x):
        #each gcnn layer is a chebyshev filter bank, relu, and pool.
        for i in range(len(self.p)):
            with tf.variable_scope('conv{}'.format(i + 1)):
                with tf.name_scope('filter'):
                    x = self.chebyshev5(x, self.L, self.F[i], self.K[i])
                with tf.name_scope('bias_relu'):
                    x = self.b1relu(x)
                with tf.name_scope('pooling'):
                    x = self.mpool1(x, self.p[i])       

        N, M, F = x.get_shape()
        x = tf.reshape(x, [int(N), int(M*F)])

        #self.ph_site = tf.expand_dims(self.ph_site, 1)
        #x = tf.concat([x, self.ph_site], 0)

        for i, M in enumerate(self.M[:-1]):
            with tf.variable_scope('fc{}'.format(i + 1)):
                x = tf.nn.dropout(x, self.ph_dropout)
                x = self.fc(x, M)

        # Logits linear layer
        with tf.variable_scope('logits'):
            x = tf.nn.dropout(x, self.ph_dropout)
            x = self.fc(x, self.M[-1], relu=False)

        return tf.squeeze(x)

    def run_model(self): #just one marble through the model
        next_batch = self.inputs.next_batch(1)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            loss_average = sess.run(self.model, feed_dict={"data:0": next_batch[0], "labels:0": next_batch[1], "do_rate:0": .5})
            print loss_average

    def train(self, iterations):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(iterations):
                next_batch = self.inputs.next_batch(self.batch_size)
                loss_average = sess.run(self.model, feed_dict={"data:0": next_batch[0], "labels:0": next_batch[1], "do_rate:0": .5})


if __name__ == "__main__":
    test_data = datasets.test_singles()
    #train_data = datasets.train_singles()
    coords = np.asarray([[float(cord) for cord in re.split("\n | ", line)] for line in open("1000_coords_3column.txt")])[:234] #haaack just to get the marble rolling
    dist, idx = graph.distance_scipy_spatial(coords, k=10, metric="euclidean")
    adj = graph.adjacency(dist, idx).astype(np.float64)
    L = graph.laplacian(adj, normalized=True)
    comp_graph = tf.Graph()
    with comp_graph.as_default():
        A = deep_cgnn(comp_graph, L, test_data, 1)
        A.train(100)
