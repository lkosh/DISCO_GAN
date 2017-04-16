import skimage
import numpy as np
import skimage.io as io
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import rotate
class dataloader():
	def __init__(self, path):
		self.basepath = path
		self.datapath = ''
		self.labelpath = '../../../labels/Subject1_annotations/taking_food'
		self.dirs = ['0510180218']
		self.count = [10] #[404]
		self.images = []
		self.bb = np.empty((0,4))
		self.joints = np.empty((0,45))
		self.humans = []

		self.fx = 588.03
		self.fy = 587.07
		self.ux = 320
		self.uy = 240
	def load_images(self):
		j = 0
		path = self.basepath + self.datapath
		for dir in self.dirs:
			for i in range(1,self.count[j]+1):
				im = io.imread(path + '/' + dir + '/Depth_' + str(i) + '.png')
				im = np.fliplr(rotate(im,180))
				self.images.append(im)
			j += 1

	def load_joints(self):
		path = self.basepath + self.labelpath
		j = 0

		for i in self.dirs:
			
			with open(path +'/'+ i + '.txt', 'r') as f:
				for k in range(1, self.count[j] + 1):
					line = f.readline().split(',')[:-2]
					
					line = map(float,line)
					
					#line = np.array(line)
					joints=[]
					joints.append(line[11:14])#1
					joints.append(line[25:28])#2
					joints.append(line[39:42])#3
					joints.append(line[53:56])#4
					joints.append(line[67:70])#5
					joints.append(line[81:84])#6
					joints.append(line[95:98])#7
					joints.append(line[109:112])#8
					joints.append(line[123:126])#9
					joints.append(line[137:140])#10
					joints.append(line[151:154])#11
					joints.append(line[155:158])#12
					joints.append(line[159:162])#13
					joints.append(line[163:166])#14
					joints.append(line[167:171])#15
					
					joints = np.array(joints).ravel()
					joints = np.reshape(joints, (1,45))
					self.joints = np.vstack((self.joints,joints))

			j += 1
					
	def load_bb(self, obj_id):
		path = self.basepath+self.labelpath
		j = 0
		for i in self.dirs:
			with open(path + '/' + i + "_obj" + obj_id + ".txt") as f:
				for k in range(1, self.count[j] +1):
					bb = []
					line = f.readline().split(",")[:-2]
					#print line
					line = map(float, line)
					bb.append(line[2:6])
					self.bb = np.vstack((self.bb, bb))
			j += 1
		#self.bb = bb
	def cut_human(self):
		j = 0
		for i in self.dirs:
			
			for k in range(1, self.count[j] ):
				#print self.joints[k, 0::3]
				x1 = np.min(self.joints[k,0::2]) + self.images[k].shape[1]/2
				y1 = np.min(self.joints[k,1::2]) + self.images[k].shape[0]/2
				x2 = np.max(self.joints[k,2::2]) + self.images[k].shape[1]/2
				y2 = np.max(self.joints[k,3::2]) + self.images[k].shape[0]/2
				sample1 = [x1,y1,0]
				print x1,x2,y1,y2
				x1,y1,z = self.joint3DToImg(sample1)
				sample2 = [x2,y2,0]
				x2, y2,z  = self.joint3DToImg(sample2)
				#self.humans.append(self.images[k][x1:x2, y1:y2])
				print x1,x2,y1,y2
				#io.imshow(self.humans[k-1])
				#io.show()
			j += 1

	def joints_to_uvd(self):
		joints_uvd = np.zeros((self.joints.shape), dtype = float)
		for j in range(0,self.joints.shape[0]):
			array  = self.joints[j,:]
			for i in range(0, array.shape[0], 3):
				x,y,z = array[i:i+3]
				u,v,d = self.joint3DToImg([x,y,z])
				joints_uvd[j,i:i+3] = u,v,d

		return joints_uvd
	def load_depth(self):
		images = []
		for root, dirs, files in os.walk(self.basepath):
			path = root.split(os.sep)
			print((len(path) - 1) * '---', os.path.basename(root))
			for file in files:
				format = file.split('.')[1]
				type = file.split('_')[0]
				if format == 'png' and type == 'Depth':
					print format, type, file, path
					im = io.imread(self.basepath +  path + file)
					images.append(im)

	def joint3DToImg(self, sample):
        
		#Denormalize sample from metric 3D to image coordinates
		#:param sample: joints in (x,y,z) with x,y and z in mm
		#:return: joints in (x,y,z) with x,y in image coordinates and z in mm
		
		ret = np.zeros((3,), np.float32)
        # convert to metric using f
		if sample[2] == 0.:
			ret[0] = self.ux
			ret[1] = self.uy
			return ret
		ret[0] = sample[0]/sample[2]*self.fx+self.ux
		ret[1] = sample[1]/sample[2]*self.fy+self.uy
		ret[2] = sample[2]
		return ret	

 

#loader.load_depth()
import numpy as np
import time
import sys
###################### ADD PATH TO THE DEEP PRIOR PACKAGE HERE
sys.path.append('../../utils/')

import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import cuda
from chainer import serializers
import cupy
from scores import *

n_conv = 5
n_filters = 8
n_pool = 3
n_pixels_1 = 128
n_pixels_2 = 128
size_out_1 = 48 #9
size_out_2 = 66 #9

class JointPositionExtractor(chainer.Chain):
	
	def __init__(self, score, nrand, J = 14):
		super(JointPositionExtractor, self).__init__(
			# image network
			conv0 = L.Convolution2D(1, n_filters, n_conv, stride = 1, pad=0, wscale=0.01*math.sqrt(n_conv*n_conv)),
			conv1 = L.Convolution2D(n_filters,n_filters,n_conv,stride = 1, pad=0, wscale=0.01*math.sqrt(n_conv*n_conv*n_filters)),
			conv2 = L.Convolution2D(n_filters,n_filters,n_conv,stride = 1, pad=0, wscale=0.01*math.sqrt(n_conv*n_conv*n_filters)),
			
			# start concatenating
			lin2 = L.Linear(size_out_1*size_out_2*n_filters + nrand, 1024, wscale=0.01*math.sqrt(size_out_1*size_out_2*n_filters + nrand)),
			lin3 = L.Linear(1024, 1024, wscale=0.01*math.sqrt(1024)),
			lin4 = L.Linear(1024, 3*J, wscale=0.01*math.sqrt(1024)),

		)
		self.es = None
		#number of joints
		self.J = J
		self.nrand = nrand
		self.score = score
		self.n_per_sample = self.score.n_per_sample
	# get features, wxtracted by NN 
	def fast_sample_depth_image(self, x, test=False):
	#h_image = x -> convo -> relu -> maxpooling -> convo -> relu -> pooling -> convo -> relu -> (y, 9*9*8) - 2d array,
	# result of feeding an image x to the NN
		h_image = F.max_pooling_2d(F.relu(self.conv0(x)), ksize = n_pool)
		h_image = F.max_pooling_2d(F.relu(self.conv1(h_image)), ksize = n_pool)
		h_image = F.relu(self.conv2(h_image))
		print h_image.shape
		h_image = F.reshape(h_image, (h_image.data.shape[0], size_out_1*size_out_2*n_filters))
		return h_image

	def fast_sample(self, z, image_features, test=False):
		h = F.concat((z,image_features), axis = 1)
		h = F.relu(self.lin2(h))
		h = F.relu(self.lin3(h))
		h = self.lin4(h)
		return h

	def __call__(self, x, y, test=False):

		features = self.fast_sample_depth_image(x)
		z = chainer.Variable(xp.random.uniform(-1.0, 1.0,
		(x.data.shape[0], self.nrand)).astype(xp.float32)) #noise
		Y = self.fast_sample(z, features) #pose estimation for (x_n, z_n)
		Y = F.expand_dims(Y, 1)
		Y_mul = F.concat((Y,))
		
		for k in range(self.n_per_sample - 1):
			z = chainer.Variable(xp.random.uniform(-1.0, 1.0,
			(x.data.shape[0], self.nrand)).astype(xp.float32))
			Y = self.fast_sample(z, features)
			Y = F.expand_dims(Y, 1)
			Y_mul = F.concat((Y_mul, Y), axis = 1)

		es = compute_score(self.score, Y_mul, y) # call scoring function
		self.es = es.data
		return es

def weight_on_fingers(fingers, weight):
	
	joints = ["Pinky tip", "Pinky mid", "Ring tip", "Ring mid", "Middle tip", "Middle mid", "Index tip", "Index mid", \
		"Thumb tip", "Thumb mid", "Thumb root", "Palm left", "Palm right", "Palm"]
	J = len(joints)
	# array of weights - 14 x 3
	weighting =  weight * xp.ones((J,3)).astype(xp.float32)
	for j in range(J):
		#finger_name - Pinky, Ring , etc
		finger_name = joints[j].split(" ")[0]
		if finger_name in fingers:
			weighting[j,:] = 1.
	weighting = xp.reshape(weighting, 3*J)
	return weighting

def objective_function(model, x_val, y_val, n_per_sample_val, z_monitor_objective_all_val):
	
	N_val = x_val.shape[0]
	start = 0
	end = 0
	batchsize = max(N_val,600)
	loss = 0.0
	while True:
		start = end
		end = min([start+batchsize,N_val])
		x = chainer.Variable(x_val[start:end,:,:,:].astype(xp.float32))
		y = chainer.Variable(y_val[start:end,:])
		ex = model.fast_sample_depth_image(x)
		z = chainer.Variable(xp.asarray(z_monitor_objective_all_val[start:end, 0, :]))
		pred = model.fast_sample(z,ex)
		pred = F.expand_dims(pred, axis = 1)
		preds = F.concat((pred,))
		for k in range(n_per_sample_val - 1):
			z = chainer.Variable(xp.asarray(z_monitor_objective_all_val[start:end, k+1, :]))
			pred = model.fast_sample(z,ex)
			pred = F.expand_dims(pred, axis = 1)
			preds = F.concat((preds,pred), axis = 1)
		loss += model.score(preds,y).data * (end - start)
		if end == N_val:
			break
	return loss / N_val
	
if __name__ == '__main__':
	gpu_id = float(sys.argv[1])	
	#setting parameters
	beta = 1.0
	seed = 0
	alpha = 0.5
	C = 1e-3
	savedir = './HumanTryRelease'
	nrand = 200
	finger_w = 1.0
	fingers = ["Pinky,Ring,Middle,Palm,Index,Thumb"]

	n_per_sample = 2
	if alpha > 0.0:
		n_per_sample_val = 2
	else:
		n_per_sample_val = 1
	
	tol = 1e-12
	J = 15
	loader = dataloader('data/Subject1_rgbd_images/taking_food/')
	loader.load_images()
	#loader.load_bb("2")
	#print len(loader.images)
	loader.load_joints()
	print loader.joints.shape
	#x1,y1,x2,y2 = loader.bb[0]
	#loader.cut_human()
	
	#uvd_joints = loader.joints_to_uvd()
	#print uvd_joints[0], loader.joints[0]
	#for j in range(len(loader.images)):
#		fig,ax = plt.subplots(1)
	#	im = loader.images[j]
	#	ax.imshow(im)
		#rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,edgecolor='r',facecolor='none')

	#	for i in range(0, 15):
	#		circ = patches.Circle((uvd_joints[j][0+3*i],uvd_joints[j][1+3*i]),5)
	#		ax.add_patch(circ)
	#	plt.show()

	
	# this uses code of deepprior repo, specifically - data preparation (src/data)
	# need to replace this later with smth that prepares my dataset the same way

#X_train is probably a stack of images , Y_train - 14 joints for each image
	use_gpu = True
	#gpu_id = 2
	
	if use_gpu:
		cuda.get_device(gpu_id).use()
		xp = cuda.cupy
	else:
		xp = np
	
#	with cupy.cuda.Device(gpu_id):
	X_train = xp.asarray(loader.images)
	X_train = xp.reshape(X_train, (X_train.shape[0],1, X_train.shape[1], X_train.shape[2]))
	Y_train = xp.asarray(loader.joints)
#	print X_train.device
	
	# Y.shape = (n_samples, 14, 1) ? --> (n_samples, 14)
	#Y_train = xp.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1]* Y_train.shape[2]))
	
	np.random.seed(0)
	# shuffle the training samples
	indexes = np.arange(X_train.shape[0])
	np.random.shuffle(indexes)
	Nval = int(X_train.shape[0] * 0.2)

	#Nval = 100 # save 10000 for validation set
	indexes_val = indexes[:Nval]
	indexes_train = indexes[Nval:]
	#with cuda.Device(gpu_id):
	x_val = xp.asarray(X_train[indexes_val,:,:,:]).astype(xp.float32)
	y_val = xp.asarray(Y_train[indexes_val,:]).astype(xp.float32)
	x_train = X_train[indexes_train,:,:,:]
	y_train = Y_train[indexes_train,:]

	N = x_train.shape[0]
	#Create random noise to evaluate current objective function
	z_monitor_all_val = xp.asarray(np.random.uniform(-1.0, 1.0,
						(Nval, n_per_sample_val, nrand)).astype(np.float32))

	np.random.seed(seed)
	# score function 
	#object from utils
	scoring = Score(beta, alpha, n_per_sample)
	#if weight != 1 it means that some fingers positions are more crucial than others 
	model = JointPositionExtractor(scoring, nrand, J)
	if use_gpu:
		model.to_gpu(gpu_id)
	#chainer object
	opt_model = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
	
	opt_model.setup(model)
	opt_model.add_hook(chainer.optimizer.WeightDecay(C))
	
	batchsize = min(N, 128)
	size_epoch = int(N/batchsize) + 1
	monitor_frequency = 10
	Nepoch = 200
#	with cuda.Device(gpu_id):

	Probloss_val = xp.asarray(np.zeros(Nepoch))
	Probloss_train = xp.asarray(np.zeros(Nepoch))
	start_at = time.time()
	MER_val = xp.zeros((Nepoch))
 	print "Starting training..."
	with cupy.cuda.Device(gpu_id):
		epoch = 0
		period_start_at = time.time()
		bi = 0
		curr_epoch = 0
		while True:
			#monitor objective value
			if bi % size_epoch == 0:
				if curr_epoch % monitor_frequency == 0 or curr_epoch == (Nepoch-1):
					serializers.save_npz(savedir + '/model_%d.model' % curr_epoch, model) # save model every epoch
					MER_val[curr_epoch] = objective_function(model, x_val, y_val, n_per_sample_val, z_monitor_all_val)
					now = time.time()
					tput = float(size_epoch*monitor_frequency*batchsize) / (now-period_start_at)
					tpassed = now-start_at
					print "   %.1fs Epoch %d, batch %d, Probloss on Validation Set %.4f, %.2f S/s" % \
						(tpassed, curr_epoch, bi, MER_val[curr_epoch],tput)
					# Reset
					period_start_at = time.time()
				
				curr_epoch += 1
				if curr_epoch >= Nepoch:
					print("we're stopping")
					break
			bi += 1  # Batch index
			indexes = np.sort(np.random.choice(N, batchsize, replace=False))
			x = chainer.Variable(xp.asarray(x_train[indexes]).astype(xp.float32))
			y = chainer.Variable(xp.asarray(y_train[indexes]).astype(xp.float32))   
			# Reset/forward/backward/update
			opt_model.update(model, x, y)


		serializers.save_npz(savedir + '/model_end.model', model)
		#np.savetxt(savedir + 'objective_values', [objective_values])
		np.savetxt(savedir + 'MER_val', [MER_val])
		serializers.save_npz(savedir + 'optimizer_end.state', opt_model)
