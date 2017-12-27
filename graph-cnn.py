

#libraries
import models, graph, coarsening, utils
import numpy as np
import matplotlib.pyplot as plt
import time 
import tensorflow as tf
import math
import os 
import glob
from numpy import genfromtxt
import pdb
import scipy
from sklearn.cross_validation import train_test_split
from scipy import sparse
import os
from hostlist import expand_hostlist

#######server things/start
'''
task_index = int(os.environ['SLURM_PROCID'])
task_index = 1;
n_tasks = int(os.environ['SLURM_NPROCS'])
tf_hostlist = [("%s:2222" % host) for host in expand_hostlist(os.environ['SLURM_NODELIST']) ]
print(tf_hostlist)

cluster = tf.train.ClusterSpec({"taskname1" : tf_hostlist})
server  = tf.train.Server( cluster,job_name = "taskname1",task_index =task_index)

'''
cluster = tf.train.ClusterSpec({"local":["wbic-cs-1:12000","wbic-cs-2:12001"]})
server  = tf.train.Server( cluster,job_name = "local",task_index =0)
server  = tf.train.Server( cluster,job_name = "local",task_index =1)
######server things/end

#read data
def read_mat_stuff(N,per_subj,directory):
    counter = 0
    my_data = [None]*per_subj*N*2
    for fname in glob.glob(directory):
      my_data[counter] = genfromtxt(fname, delimiter=',')
      counter = counter +1
    return my_data     

#read data for perm that comes from coarsening (see paper)
def read_mat_stuff_perm(N,per_subj,directory,perm): 
    counter = 0
    my_data_perm = [None]*per_subj*N*2
    for fname in glob.glob(directory):
      my_data = genfromtxt(fname, delimiter=',')
      my_data_perm[counter] = coarsening.perm_data(my_data,perm)
      counter = counter +1
    return my_data_perm    

#sparse mat to sparse tensor
def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row,coo.col]).transpose() 
    return tf.SparseTensor(indices,coo.data,coo.shape)


#read labels
def read_labels(file_):
    my_label = genfromtxt(file_, delimiter=',')
    return my_label  

#read distance matrix (assuming same for all samples for now...)
def read_A(file_):
    A = genfromtxt(file_,delimiter=',')
    return A	    

#create pairs
def create_pairs(data,indices,N):
    pairs = []
    labels = []
    
    for i in range(N):
        for j in range(N):
            if indices[i] == indices[j]:
               pairs += [[data[i],data[j]]]
               labels += [1]
            else:   
               pairs += [[data[i],data[j]]]
               labels += [0] 
    return np.array(pairs), np.array(labels)


#creating a random Laplacian; no relationship with our data...
'''
def make_laplacian(A): 
    #dum stuff: 
    d =115
    n = 1000
    c = 5
    levs = 3
    X = np.random.normal(0,1,(n,d)).astype(np.float32)
    X += np.linspace(0,1,c).repeat(d // c)


    dist, idx = graph.distance_scipy_spatial(X.T, k=10, metric='euclidean')
    A = graph.adjacency(dist, idx).astype(np.float32)
    graphs, perm = coarsening.coarsen(A, levels=levs, self_connections=False)
    L_list = [graph.laplacian(A, normalized=True) for A in graphs]
    L = L_list[0].astype(np.float32)

    return L, perm, levs, L_list
'''


#read some distance matrix (e.g. from xyz coordinates) in order to make Laplacian; this needs to be precomputed and lying somewhere in your hard drive
def make_laplacian_directory(fname_input): 
    levs = 3
    A = genfromtxt(fname_input, delimiter=',')
    A_ = sparse.csr_matrix(A).astype(np.float32)
    
    graphs, perm = coarsening.coarsen(A_, levels=levs, self_connections=False)
    L_list = [graph.laplacian(g, normalized=True) for g in graphs]
    L = L_list[0].astype(np.float32)
    #pdb.set_trace()
    return L, perm, levs, L_list

'''
#weight variables
def _weight_variable(shape,regularization=True):
 
    initial = tf.truncated_normal_initializer(0,0.1)
    var = tf.get_variable('weights',shape, tf.float32, initializer=initial)
#    if regularization: 
#       regularizers.append(tf.nn.12_loss(var))
#    tf.summary.histogram(var.op.name,var)
    return var

#bias variables
def _bias_variable(shape,regularization=True):

    initial = tf.constant_initializer(0.1)
    var = tf.get_variable('bias',shape, tf.float32, initializer=initial)
#    if regularization: 
#       regularizers.append(tf.nn.12_loss(var))
#    tf.summary.histogram(var.op.name,var)
    return var
'''



#weight variables
def _weight_variable(shape,regularization=True):
# with tf.device(tf.train.replica_device_setter(worker_device="/job:taskname1/task:0",cluster = cluster)):
    initial = tf.truncated_normal_initializer(0,0.1)
    var = tf.get_variable('weights',shape, tf.float32, initializer=initial)
#    if regularization: 
#       regularizers.append(tf.nn.12_loss(var))
#    tf.summary.histogram(var.op.name,var)
    return var

#bias variables
def _bias_variable(shape,regularization=True):
# with tf.device(tf.train.replica_device_setter(worker_device="/job:taskname1/task:1",cluster = cluster)):
    initial = tf.constant_initializer(0.1)
    var = tf.get_variable('bias',shape, tf.float32, initializer=initial)
#    if regularization: 
#       regularizers.append(tf.nn.12_loss(var))
#    tf.summary.histogram(var.op.name,var)
    return var




#fully connected layer
def fc_other(x,Mout,relu=True):
    N,Min = x.get_shape()
    W = _weight_variable([int(Min),Mout],regularization=True)
    b = _bias_variable([Mout],regularization=True)
    x = tf.matmul(x,W)+b
    return tf.nn.relu(x) if relu else x

#filter fourier
def filter_in_fourier(x,L,Fout,K,U,W):
    N,M,Fin = x.get_shape()
    N,M,Fin = int(N),int(M),int(Fin)
    x = tf.transpose(x,perm=[1,2,0])
    
    x = tf.reshape(x,[M,Fin*N])
    x = tf.matmul(U,x)
    x = tf.reshape(x,[M,Fin,N])

    x = tf.matmul(W,x)
    x = tf.transpose(x)
    x = tf.reshape(x,[N*Fout,M])
    
    x = tf.matmul(x,U)
    x = tf.reshape(x,[N,Fout,M])
    return tf.transpose(x,perm=[0,2,1])

#fourier
def fourier(x,L,Fout,K):
    assert K == L.shape[O]
    N,M,Fin = x.get_shape()
    N,M,Fin = int(N),int(M),int(Fin)
  
    _,U = graph.fourier(L)
    U = tf.constant(U.T,dtype = tf.float32)
 
    W = _weight_variable([M,Fout,Fin],regularization=False)
    return filter_in_fourier(x,L,Fout,K,U,W)

#bspline basis
def bspline_basis(K,x, degree =3):
    if np.isscalar(x):  
       x = np.linspace(0,1,x)
    kv1 = x.min() * np.ones(degree)
    kv2 = np.linspace(x.min(),x.max(),K-degree+1)
    kv3 = x.max() * np.ones(degree)
    kv = np.concatenate((kv1,kv2,kv3))

#spline
def spline(x,L,Fout,K):
    N,M,Fin = x.get_shape()
    N,M,Fin = int(N),int(M),int(Fin)

    lamb, U = graph.fourier(L)
    U = tf.constant(U.T,dtype =tf.float32)

    B = bspline_basis(K,lamb,degree=3)
    B = tf.constant(B,dtype = tf.float32)

    W = _weight_variable([K,Fout*Fin],regularization=False)
    W = tf.matmul(B,W)
    W = tf.reshape(W,[M,Fout,Fin])
    return filter_in_fourier(x,L,Fout,K,U,W)

#chebyshev2
def chebyshev2(x,L,Fout,K,name='cheby2'):
 with tf.variable_scope(name):
   
    _,M,Fin = x.get_shape()
    M,Fin = int(M),int(Fin)
    L = scipy.sparse.csr_matrix(L)
    L = graph.rescale_L(L,lmax =2)

    x = tf.transpose(x,perm=[1,2,0])
    #x = tf.reshape(x,[M,Fin*N])
    x = tf.reshape(x,[M,-1])
    def chebyshev(x):
        return graph.chebyshev(L,x,K)
    x = tf.py_func(chebyshev,[x],[tf.float32])[0]
    #x = tf.reshape(x,[K,M,Fin,N])
    x = tf.reshape(x,[K,M,Fin,-1])
    x = tf.transpose(x,perm=[3,1,2,0])
    #x = tf.reshape(x,[N*M,Fin*K])
    x = tf.reshape(x,[-1,Fin*K])
    
    W = _weight_variable([Fin*K,Fout],regularization=True)
    x = tf.matmul(x,W)
    
    #return tf.reshape(x,[N,M,Fout])
    
    return tf.reshape(x,[-1,M,Fout])

#chebyshev5
def chebyshev5(x,L,Fout,K,name='cheby'):
 with tf.variable_scope(name):
    list_size = tf.shape(x) #this is for the None shape of the Tensor...
    N = list_size[0]
    _,M,Fin = x.get_shape()
    #pdb.set_trace()
    M,Fin = int(M),int(Fin)
    # Rescale Laplacian and store as sparse Tensor
    
    L = scipy.sparse.csr_matrix(L)
    L = graph.rescale_L(L,lmax = 2)
    L = L.tocoo()
    indices = np.column_stack((L.row,L.col))
    L = tf.SparseTensor(indices,L.data,L.shape)
    L = tf.sparse_reorder(L)

    x0 = tf.transpose(x,perm=[1,2,0])   # M x Fin x N
    x0 = tf.reshape(x0,[M,Fin*N])       # M x Fin*N
    x = tf.expand_dims(x0,0)		# 1 x M x Fin*N

    def concat(x,x_):
        x_ = tf.expand_dims(x_,0)	# 1 x M x Fin*N
        return tf.concat([x,x_],axis =0)# K x M x Fin*N
     
    #pdb.set_trace()
    if K > 1:
        x1 = tf.sparse_tensor_dense_matmul(L,x0)
        x = concat(x,x1)
    for k in range(2,K):
        x2 = 2*tf.sparse_tensor_dense_matmul(L,x1) - x0 # M x Fin*N
        x = concat(x,x2)
        x0,x1 = x1,x2
    x = tf.reshape(x,[K,M,Fin,N])	# K x M x Fin x N
    x = tf.transpose(x,perm=[3,1,2,0])  # N x M x Fin x K
    x = tf.reshape(x,[N*M,Fin*K])	# N*M x Fin*K
    W = _weight_variable([Fin*K,Fout],regularization =True)
    x = tf.matmul(x,W) 			# N*M x Fout
    return tf.reshape(x,[N,M,Fout]) 	# N x M x Fout

#bias and ReLU
def b1relu(x,name='reli'):
  with tf.variable_scope(name):
    N,M,F = x.get_shape()
    b = _bias_variable([1,1,int(F)],regularization=False)
    return tf.nn.relu(x+b)

#bias and ReLU 2
def b2relu(x, name ='reli2'):
   with tf.variable_scope(name):
    N,M,F = x_get_shape()
    b = _bias_variable([1,int(M),int(F)],regularization =False)
    return tf.nn.relu(x+b)

#max pooling
def mpool1(x,p,name='pooli'):
 with tf.variable_scope(name):
    if p > 1 :
         x = tf.expand_dims(x,3) 	# N x M x F x 1
         x = tf.nn.max_pool(x,ksize=[1,p,1,1],strides=[1,p,1,1],padding ='SAME')
         return tf.squeeze(x,[3])	# N x M/p x F
    else:
         return x
#max pooling 2
def apool1(x,p, name='pooli2'):
 with tf.variable_scope(name): 
    if p >1 :
         x = tf.expand_dims(x,3)
         x = tf.nn.avg_pool(x,ksize=[1,p,1,1],strides=[1,p,1,1],padding = 'SAME')
         return tf.squeeze(x,[3])
    else: 
         return x





#############model/start 
def layers2(x,dropout,L,L1,L2,L3):
    #params
    F1 = 32
    F2 = 32
    K = 20
    ROIS = 118
    p1 = 4
    p2 = 2
    NCLASSES = 1
    ###
    shape_ = x.get_shape().as_list()
    s1 = shape_[0]  #batch_size (this is of None type)
    s2 = shape_[1]  #ROIS
    s3 = shape_[2]  #features (M)
    #x = tf.reshape(x,[s3,s1*s2])  
    x = tf.reshape(x,[s3,-1]) 
    x = tf.transpose(x)
    x = tf.expand_dims(x,2)  #samples(N) x M x F(=1)

    # apply cheby
    #with tf.device("/job:local/task:0"):
    out1 = chebyshev2(x,L,F1,K,name='cheby1')   
    # apply bias and ReLU
    #with tf.device("/job:local/task:0"):
    out2 = b1relu(out1,name='reli1')
    # apply max pooling
    out3 = mpool1(out2,p1,name='pooli1')

    

    #continue with second level...
 
    #out4 = chebyshev5(out3,L2,F2,K,name='cheby2')
    
    #out5 = b1relu(out4,name='reli2')
    
    #out6 = mpool1(out5,p2,name='pooli2')    
    
    
    #fully connected with dropout
    out6 = out3 #hard hack skipping out4,out5
    N,F,M = out6.get_shape() 			# N x M x F
    
    F = int(F)
    M = int(M)
    
    out7= tf.reshape(out6,[-1,ROIS,F,M])        # N_pair x ROIs x F x M
    W = _weight_variable([ROIS*F*M,NCLASSES],regularization=True)
    b = _bias_variable([NCLASSES],regularization=True)
    y = tf.reshape(out7,[-1,ROIS*F*M])    # N_pairs x 1
    y =  tf.matmul(y,W) + b
    #y = tf.nn.dropout(y,dropout)
    
    
    return y


###############model/end   
    
#build model    
def build_model_new(x,dropout,L,L1,L2,L3):
    m1 = layers2(x,dropout,L,L1,L2,L3)
    return m1    

#loss function
def contrastive_loss(y,d):
    tmp= y *tf.square(d)
    #tmp= tf.mul(y,tf.square(d))
    tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
    return tf.reduce_sum(tmp +tmp2)/batch_size/2

#accuracy
def compute_accuracy(prediction,labels):
    return labels[prediction.ravel() < 0.5].mean()
    #return tf.reduce_mean(labels[prediction.ravel() < 0.5])

#next batch
def next_batch(s,e,inputs,labels):
    input1 = inputs[s:e,0]
    input2 = inputs[s:e,1]
    y= np.reshape(labels[s:e],(len(range(s,e)),1))
    return input1,input2,y

######
# Initializing the variables
init = tf.initialize_all_variables()

#number of subjects
N = 21
#number of matrices per subject (coming from dynamical conn)
per_subj = 20
N = int(N)
per_subj = int(per_subj)

#common A (read it for now)

#A = read_A("/lustre/scratch/wbic-beta/mmc57/hcp_data/sst_rest_data/csv_files/laplacian/A.csv")
fname_ = "/lustre/scratch/wbic-beta/mmc57/hcp_data/sst_rest_data/csv_files/laplacian/dum_Lapl.csv"

#make Laplacian
#L, perm, levs, L_list = make_laplacian(A)
L, perm, levs, L_list = make_laplacian_directory(fname_)
_, U = graph.fourier(L)
U = U.astype(np.float32)
L1 = L_list[1]
L2 = L_list[2]
L3 = L_list[3]


#L = convert_sparse_matrix_to_sparse_tensor(L)
#L1 = convert_sparse_matrix_to_sparse_tensor(L[1])
#L2 = convert_sparse_matrix_to_sparse_tensor(L[2])
#L3 = convert_sparse_matrix_to_sparse_tensor(L[3])



directory1 ="/lustre/scratch/wbic-beta/mmc57/hcp_data/sst_rest_data/csv_files/mat_files3/*.csv"
data = read_mat_stuff_perm(N,per_subj,directory1,perm)
directory2 = "/lustre/scratch/wbic-beta/mmc57/hcp_data/sst_rest_data/csv_files/indices3/label.csv"
input_labels = read_labels(directory2)




acc_table = np.empty(2)



#1st split
print('random split %d' % 1)
global_step = tf.Variable(0,trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,10,0.1,staircase=True)
# the data: split between train and test sets
const_ = N*per_subj*2 # this how many I have for now...


#X_train, X_test,ind_train, ind_test = train_test_split(data,input_labels,test_size =0.2, random_state = 42)

X_train_val, X_test,ind_train_val, ind_test = train_test_split(data,input_labels,test_size = 0.2, random_state = 42)
X_train, X_val, ind_train, ind_val = train_test_split(X_train_val,ind_train_val,test_size = 0.4, random_state = 42)

n_train = int(len(X_train))
n_val = int(len(X_val))
n_test = int(len(X_test))

# create training, validation, and test pairs
tr_pairs, tr_y = create_pairs(X_train, ind_train, n_train) 

tv_pairs, tv_y = create_pairs(X_val, ind_val, n_val) 
te_pairs, te_y = create_pairs(X_test, ind_test, n_test) 
#pdb.set_trace()
# some placeholders
shape_ = L.shape[0]  #initial number of features = number of coarsened nodes
shape_1 = 118;       #ROIs
batch_size =	30
n_epochs = 100
matrices_L = tf.placeholder(tf.float32,shape=([None,shape_1,shape_]),name='L')
matrices_R = tf.placeholder(tf.float32,shape=([None,shape_1,shape_]),name='R')
labels = tf.placeholder(tf.float32,shape =([None,1]),name='gt')
dropout_f = tf.placeholder("float")





with tf.variable_scope("siamese") as scope:
    model1 = build_model_new(matrices_L,dropout_f,L,L1,L2,L3)
    scope.reuse_variables()
    model2 = build_model_new(matrices_R,dropout_f,L,L1,L2,L3)

#contrastive loss
distance  = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(model1,model2),2),1,keep_dims=True))
loss = contrastive_loss(labels,distance)


#t_vars = tf.trainable_variables()
#d_vars  = [var for var in t_vars if 'l' in var.name]

batch = tf.Variable(0)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss)
#optimizer = tf.train.RMSPropOptimizer(0.0001,momentum=0.9,epsilon=1e-6).minimize(loss)

#some thing for the plotting
accuracy_keep = np.empty(n_epochs)
loss_keep = np.empty(n_epochs)

# Launch the graph

with tf.Session() as sess:
#with tf.Session("grpc://localhost:12000") as sess:
    #sess.run(init)
    tf.initialize_all_variables().run()
    # Training cycle
    for epoch in range(n_epochs):
        avg_loss = 0.
        avg_acc = 0.
        #total_batch = int(X_train.shape[0]/batch_size)
        total_batch = int(n_train/batch_size)
        start_time = time.time()
        # Loop over all batches
        for i in range(total_batch):
            s  = i * batch_size
            e = (i+1) *batch_size
            # Fit training using batch data
            input1,input2,y =next_batch(s,e,tr_pairs,tr_y)
            input1_v,input2_v,y_v =next_batch(s,e,tv_pairs,tv_y) 
	    #with tf.device("/job:local/task:1"):
            _,loss_value,predict=sess.run([optimizer,loss,distance], feed_dict={matrices_L:input1,matrices_R:input2 ,labels:y,dropout_f:0.2})
            feature1=model1.eval(feed_dict={matrices_L:input1,dropout_f:0.2})
            feature2=model2.eval(feed_dict={matrices_R:input2,dropout_f:0.2})
            _,loss_value_v,predict_v=sess.run([optimizer,loss,distance], feed_dict={matrices_L:input1_v,matrices_R:input2_v ,labels:y_v,dropout_f:0.2})
            feature1_v=model1.eval(feed_dict={matrices_L:input1_v,dropout_f:0.2})
            feature2_v=model2.eval(feed_dict={matrices_R:input2_v,dropout_f:0.2})

            #pdb.set_trace()
            tr_acc = compute_accuracy(predict,y)
            tv_acc = compute_accuracy(predict_v,y_v) 
            if math.isnan(tr_acc) and epoch != 0 and math.isnan(val_acc):
                print('tv_acc %0.2f' % tv_acc)
                #pdb.set_trace()
            avg_loss += loss_value_v
            avg_acc +=tv_acc*100
        #print('epoch %d loss %0.2f' %(epoch,avg_loss/total_batch))
        
        loss_keep[epoch] = avg_loss/total_batch
        duration = time.time() - start_time
        print('epoch %d  time: %f loss %0.5f acc %0.2f' %(epoch,duration,avg_loss/(total_batch),avg_acc/total_batch))
        accuracy_keep[epoch] = avg_acc/total_batch
    
    fig, ax1 = plt.subplots(figsize=(15,5))
    ax1.plot(accuracy_keep,'b.-')
    ax1.set_ylabel('accuracy',color = 'b')
    ax2 = ax1.twinx()
    ax2.plot(loss_keep, 'g.-')
    ax2.set_ylabel('training loss',color = 'g') 
    plt.show()
    np.savetxt('split1_acc.txt',accuracy_keep,delimiter=',')
    np.savetxt('split1_loss.txt',loss_keep,delimiter=',')
    
    # Validation model
    #y = np.reshape(tv_y,(tv_y.shape[0],1))
    #pdb.set_trace()
    #predict=distance.eval(feed_dict={matrices_L:tv_pairs[:,0],matrices_R:tv_pairs[:,1],labels:y,dropout_f:0.2})
    #tr_acc = compute_accuracy(predict,y)
    #print('Accuracy validation set %0.2f' % (100 * tr_acc))

    # Test model
    total_batch = int(n_test/batch_size)
    avg_acc = 0
    for i in range(total_batch):
            s  = i * batch_size
            e = (i+1) *batch_size
            # Fit test using batch data
            input1,input2,y =next_batch(s,e,te_pairs,te_y)
            predict=distance.eval(feed_dict={matrices_L:input1,matrices_R:input2,labels:y,dropout_f:0.2})
            y = np.reshape(te_y,(te_y.shape[0],1))
            te_acc = compute_accuracy(predict,y)
            avg_acc +=te_acc*100

    print('test acc')
    print(avg_acc/total_batch) 


  
    
    



'''

###################################################################

###################################################################

#2nd split
print('random split %d' % 2)
global_step = tf.Variable(0,trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,10,0.1,staircase=True)
# the data: split between train and test sets
const_ = N*per_subj*2 # this how many I have for now...


#X_train, X_test,ind_train, ind_test = train_test_split(data,input_labels,test_size =0.2, random_state = 42)

X_train_val, X_test,ind_train_val, ind_test = train_test_split(data,input_labels,test_size = 0.2, random_state = 42)
X_train, X_val, ind_train, ind_val = train_test_split(X_train_val,ind_train_val,test_size = 0.4, random_state = 42)

n_train = int(len(X_train))
n_val = int(len(X_val))
n_test = int(len(X_test))

# create training, validation, and test pairs
tr_pairs, tr_y = create_pairs(X_train, ind_train, n_train) 

tv_pairs, tv_y = create_pairs(X_val, ind_val, n_val) 
te_pairs, te_y = create_pairs(X_test, ind_test, n_test) 

# some placeholders
shape_ = L.shape[0]  #initial number of features = number of coarsened nodes
shape_1 = 115;       #ROIs
batch_size =	30
n_epochs = 30
matrices_L = tf.placeholder(tf.float32,shape=([None,shape_1,shape_]),name='L')
matrices_R = tf.placeholder(tf.float32,shape=([None,shape_1,shape_]),name='R')
labels = tf.placeholder(tf.float32,shape =([None,1]),name='gt')
dropout_f = tf.placeholder("float")





with tf.variable_scope("siamese") as scope:
    model1 = build_model_new(matrices_L,dropout_f,L,L1,L2,L3)
    scope.reuse_variables()
    model2 = build_model_new(matrices_R,dropout_f,L,L1,L2,L3)

#contrastive loss
distance  = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(model1,model2),2),1,keep_dims=True))
loss = contrastive_loss(labels,distance)


#t_vars = tf.trainable_variables()
#d_vars  = [var for var in t_vars if 'l' in var.name]

batch = tf.Variable(0)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss)
#optimizer = tf.train.RMSPropOptimizer(0.0001,momentum=0.9,epsilon=1e-6).minimize(loss)

#some thing for the plotting
accuracy_keep = np.empty(n_epochs)
loss_keep = np.empty(n_epochs)

# Launch the graph
with tf.Session() as sess:
    #sess.run(init)
    tf.initialize_all_variables().run()
    # Training cycle
    for epoch in range(n_epochs):
        avg_loss = 0.
        avg_acc = 0.
        #total_batch = int(X_train.shape[0]/batch_size)
        total_batch = int(n_train/batch_size)
        start_time = time.time()
        # Loop over all batches
        for i in range(total_batch):
            s  = i * batch_size
            e = (i+1) *batch_size
            # Fit training using batch data
            input1,input2,y =next_batch(s,e,tr_pairs,tr_y)
            _,loss_value,predict=sess.run([optimizer,loss,distance], feed_dict={matrices_L:input1,matrices_R:input2 ,labels:y,dropout_f:0.2})
            feature1=model1.eval(feed_dict={matrices_L:input1,dropout_f:0.2})
            feature2=model2.eval(feed_dict={matrices_R:input2,dropout_f:0.2})
            #pdb.set_trace()
            tr_acc = compute_accuracy(predict,y)
            if math.isnan(tr_acc) and epoch != 0:
                print('tr_acc %0.2f' % tr_acc)
                #pdb.set_trace()
            avg_loss += loss_value
            avg_acc +=tr_acc*100
        print('epoch %d loss %0.2f' %(epoch,avg_loss/total_batch))
        
        accuracy_keep[epoch] = avg_loss/total_batch
        duration = time.time() - start_time
        print('epoch %d  time: %f loss %0.5f acc %0.2f' %(epoch,duration,avg_loss/(total_batch),avg_acc/total_batch))
        accuracy_keep[epoch] = avg_acc/total_batch
    
    fig, ax1 = plt.subplots(figsize=(15,5))
    ax1.plot(accuracy_keep,'b.-')
    ax1.set_ylabel('accuracy',color = 'b')
    ax2 = ax1.twinx()
    ax2.plot(loss_keep, 'g.-')
    ax2.set_ylabel('training loss',color = 'g') 
    plt.show()
    
    # Validation model
    #y = np.reshape(tv_y,(tv_y.shape[0],1))
    #pdb.set_trace()
    #predict=distance.eval(feed_dict={matrices_L:tv_pairs[:,0],matrices_R:tv_pairs[:,1],labels:y,dropout_f:0.2})
    #tr_acc = compute_accuracy(predict,y)
    #print('Accuracy validation set %0.2f' % (100 * tr_acc))

    np.savetxt('split2_acc.txt',accuracy_keep,delimiter=',')
    np.savetxt('split2_loss.txt',loss_keep,delimiter=',')
    

    # Test model
    predict=distance.eval(feed_dict={matrices_L:te_pairs[:,0],matrices_R:te_pairs[:,1],labels:y,dropout_f:0.2})
    y = np.reshape(te_y,(te_y.shape[0],1))
    te_acc = compute_accuracy(predict,y)
    acc_table[1] = te_acc
    print('Accuracy test set split 2 %0.2f' % (100 * te_acc))


#print('Average test accuracy ')
#print('%0.2f' % (100 * np.mean(acc_table)))
'''
np.savetxt('all_split_acc.txt',acc_table,delimiter=',')

