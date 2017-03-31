#!/usr/bin/env python
#
# Keras GAN Implementation
# See: https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/
#
#%matplotlib inline
import os,random
os.environ["KERAS_BACKEND"] = "theano"
#os.environ["THEANO_FLAGS"]  = "device=gpu%d,lib.cnmem=0"%(random.randint(0,3))
import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle, random, sys, keras
from keras.models import Model
#from IPython import display
from keras.utils import np_utils
from tqdm import tqdm


import time
import sys
###################### ADD PATH TO THE DEEP PRIOR PACKAGE HERE
sys.path.append('../../DeepPrior/src/')
sys.path.append('../../utils/')
from data.dataset import NYUDataset
from data.importers import NYUImporter
import math
import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import cuda
from chainer import serializers
import cupy
from scores import *
from sklearn.cross_validation import train_test_split

J =14
n_conv = 5
n_filters = 8
n_pool = 3
n_pixels_1 = 128
n_pixels_2 = 128
size_out_1 = 9
size_out_2 = 9
C = 1e-4
Nepoch = 400
batchsize = 256
nrand = 200
beta = 1.0
alpha = 0.5
n_per_sample_val = 2
use_gpu = 1

def objective_function(model, x_val, y_val, n_per_sample_val, z_monitor_objective_all_val):
	
	N_val = x_val.shape[0]
	start = 0
	end = 0
	batchsize = 700
	loss = 0.0
	while True:
		start = end
		end = min([start+batchsize,N_val])
		x = chainer.Variable(x_val[start:end,:,:,:].astype(xp.float32))
		y = chainer.Variable(y_val[start:end,:])
		
		#ex = model.fast_sample_depth_image(x)
		z = chainer.Variable(xp.asarray(z_monitor_objective_all_val[start:end, 0, :]))
		pred = model(x,z)
		pred = F.expand_dims(pred, axis = 1)
		preds = F.concat((pred,))
		for k in range(n_per_sample_val - 1):
			z = chainer.Variable(xp.asarray(z_monitor_objective_all_val[start:end, k+1, :]))
			pred = model(x,z)
			pred = F.expand_dims(pred, axis = 1)
			preds = F.concat((preds,pred), axis = 1)
		loss += model.score(preds,y).data * (end - start)
		if end == N_val:
			break
	return loss / N_val

def bnorm(Z):
		Z += 1e-20 #small constant for gradient stabilitly
		norm = F.basic_math.absolute(Z)
		norm = F.basic_math.pow(norm,2.0) # here we use 2.0 as the 2nd norm parameter, hence using a strictly proper scoring rule when \alpha = 0.5
		norm = F.sum(norm, axis = -1)
		norm = F.basic_math.pow(norm, beta/2.0)
		return norm

	
class Generator(chainer.Chain):
    
    def __init__(self, score):
        super(Generator, self).__init__(
            
           # image network
			
			conv0 = L.Convolution2D(1, n_filters, n_conv, stride = 1, pad=0, wscale=0.01*math.sqrt(n_conv*n_conv)),
			conv1 = L.Convolution2D(n_filters,n_filters,n_conv,stride = 1, pad=0, wscale=0.01*math.sqrt(n_conv*n_conv*n_filters)),
			conv2 = L.Convolution2D(n_filters,n_filters,n_conv,stride = 1, pad=0, wscale=0.01*math.sqrt(n_conv*n_conv*n_filters)),
			
			# start concatenating
			lin2 = L.Linear(size_out_1*size_out_2*n_filters + nrand, 1024, wscale=0.01*math.sqrt(size_out_1*size_out_2*n_filters + nrand)),
			lin3 = L.Linear(1024, 1024, wscale=0.01*math.sqrt(1024)),
			lin4 = L.Linear(1024, 3*J, wscale=0.01*math.sqrt(1024)),
			
			bn0l = L.BatchNormalization(4*4*512),
            bn1 = L.BatchNormalization(256),
            bn2 = L.BatchNormalization(64),


        )
	self.score = score
        #Generator outputs a pose prediction
    def __call__(self, x, z):
		#z = chainer.Variable(xp.random.uniform(-1.0, 1.0,
		#(x.data.shape[0], self.nrand)).astype(xp.float32)) #noise
		
		h_image = F.max_pooling_2d(F.relu(self.conv0(x)), ksize = n_pool)
		h_image = F.max_pooling_2d(F.relu(self.conv1(h_image)), ksize = n_pool)
		h_image = F.relu(self.conv2(h_image))
		h_image = F.reshape(h_image, (h_image.data.shape[0], size_out_1*size_out_2*n_filters))
		
		d = xp.array(z.data)
		image_features = xp.array(h_image.data)
		#i_d = xp.array(image_features.data)
		#h = Variable(np.concatenate((h_image.data, prediction.data), axis = 1))
		h = F.concat((image_features, d), axis = 1)
		#h = h_image
		#h = F.concat((z,image_features), axis = 1)
		h = F.relu(self.lin2(h))
		h = F.relu(self.lin3(h))
		h = F.relu(self.lin4(h))
		
		return h
        
		

#discriminator decides whether pose prediction is from modeled or real distribution
class Discriminator(chainer.Chain):
    
    def __init__(self):
        super(Discriminator, self).__init__(

	                conv0 = L.Convolution2D(1, n_filters, n_conv, stride = 1, pad=0, wscale=0.01*math.sqrt(n_conv*n_conv)),
			conv1 = L.Convolution2D(n_filters,n_filters,n_conv,stride = 1, pad=0, wscale=0.01*math.sqrt(n_conv*n_conv*n_filters)),
			conv2 = L.Convolution2D(n_filters,n_filters,n_conv,stride = 1, pad=0, wscale=0.01*math.sqrt(n_conv*n_conv*n_filters)),
			
			lin0 = L.Linear(size_out_1*size_out_2*n_filters  + 3 * J, 200, wscale=0.01*math.sqrt(size_out_1*size_out_2*n_filters + 3*J )),
			lin1 = L.Linear(200, 200, wscale=0.01*math.sqrt(200)),
			lin2 = L.Linear(200, 1, wscale=0.01*math.sqrt(200)),
        )
        
    def __call__(self, x, prediction, test=False):
        # mattya's implementation does not have bn after c1
		h_image = F.max_pooling_2d(F.relu(self.conv0(x)), ksize = n_pool)
		h_image = F.max_pooling_2d(F.relu(self.conv1(h_image)), ksize = n_pool)
		h_image = F.relu(self.conv2(h_image))
		h_image = F.reshape(h_image, (h_image.data.shape[0], size_out_1*size_out_2*n_filters))
		
		#image_features = h_image
		#default - dropout_ration = 0.5
		#print prediction, prediction.data
		d = xp.array(prediction.data)
		image_features = xp.array(h_image.data)
		#i_d = xp.array(image_features.data)
		#h = Variable(np.concatenate((h_image.data, prediction.data), axis = 1))
		h = F.concat((image_features, d), axis = 1)
		#print h, type(h), h.shape
		h = F.relu(self.lin0(h))
		h = F.relu(self.lin1(h))
		#h = F.relu(F.Dropout(self.lin0(h)))
		#h = F.relu(F.Dropout(self.lin1(h)))
		h = self.lin2(h)
		#h = F.sigmoid(h)    
		return h


if __name__ == '__main__':
	scoring = Score(beta, alpha, n_per_sample_val)
	gen = Generator(scoring)
	dis = Discriminator()
	
	#loss = Score(beta, alpha, n_per_sample)
	

	#n_per_sample = 2
	di = NYUImporter('../../DeepPrior/data/NYU')

	Seq = di.loadSequence('test')
	trainDataset = NYUDataset([Seq])
	X_train, Y_train = trainDataset.imgStackDepthOnly('test')
	if use_gpu:
		xp = cuda.cupy
	else:
		xp = np

	Y_train = xp.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1]* Y_train.shape[2]))
	Nval = 10000
	x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.2)
	print x_train.shape	
	N,tmp,h,w = x_train.shape
	#Create random noise to evaluate current objective function
	
	np.random.seed(0)
	o_gen = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
	o_dis = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
	
	o_gen.setup(gen)
	o_dis.setup(dis)
	
	o_gen.add_hook(chainer.optimizer.WeightDecay(C))
	o_dis.add_hook(chainer.optimizer.WeightDecay(C))
	
	gen_loss_list=[]
	dis_loss_list=[]
	obj_val = xp.zeros((Nepoch))
	
	z_monitor_all_val = xp.asarray(np.random.uniform(-1.0, 1.0,
						(Nval, n_per_sample_val, nrand)).astype(np.float32))

	for epoch in range(Nepoch):
		sum_dis_loss = xp.float32(0)
		sum_gen_loss = xp.float32(0)
		#xp.random.shuffle(train_data)
		for i in range(0, N, batchsize):
			input_image = x_train[i:i+batchsize]
			n = input_image.shape[0]
			z = Variable(xp.random.uniform(-1, 1, (n, nrand)).astype(xp.float32))
			
			x = gen(input_image, z)
			yl = dis(input_image, x)
            		d = yl.data
            
			L_gen = F.sigmoid_cross_entropy(yl, Variable(np.zeros((n,1)).astype(np.int32)))
			print "L_gen",  L_gen.data
			L_dis = F.sigmoid_cross_entropy(yl, Variable(np.ones((n,1)).astype(np.int32)))
			print "L_dis", L_dis.data
			true_pose = y_train[i:i+batchsize] 
			true_pose = xp.reshape(true_pose, (true_pose.shape[0], 3 * J))                    
			true_pose = Variable(xp.asarray(true_pose).astype(xp.float32))			
			yl2 = dis(input_image, true_pose)
			

			L_dis = F.sigmoid_cross_entropy(yl2, Variable(np.zeros((n,1)).astype(np.int32)))
			o_gen.zero_grads()
			L_gen.backward()
			o_gen.update()
			
			#L_dis.zerograd()
			#o_dis.cleargrad()
			o_dis.zero_grads()
			L_dis.backward()
			o_dis.update()
			#print "grad:", min(L_gen.grad), max(L_gen.grad)         
			curr_batch_gen_loss = L_gen.data 
			sum_gen_loss += curr_batch_gen_loss
			curr_batch_dis_loss = L_dis.data
			sum_dis_loss += curr_batch_dis_loss 

			gen_loss_list.append(curr_batch_gen_loss/n)
			dis_loss_list.append(curr_batch_dis_loss/n)
			print "L_gen", L_gen.data
			print "L_dis", L_dis.data
		
			obj_val[epoch] = objective_function(gen, x_val, y_val, n_per_sample_val, z_monitor_all_val)
			print "obj val", obj_val[epoch]
			#print curr_batch_gen_loss
			#print curr_batch_dis_loss
	print 'epoch end', epoch, sum_gen_loss/N, sum_dis_loss/N
