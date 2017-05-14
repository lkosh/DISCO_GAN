import numpy as np
import sys
sys.path.append('../../utils')
sys.path.append('../../DeepPrior/src/')
from scores import *
from data.dataset import NYUDataset
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import cuda
from chainer import serializers
import cupy
from dataloader import JointPositionExtractor, dataloader
import scipy
from data.dataset import NYUDataset
from data.importers import NYUImporter
from sklearn.model_selection import train_test_split
from GAN_human  import Generator 
GAN = 1
max_y = 0
mode = 0
#np.set_printoptions(threshold='nan')
N_sampling = 100
nrand =200
gpu_id = 3
J = 15
cuda.get_device(gpu_id).use()
xp = cuda.cupy
def l2norm(Z, beta = 1.0): # the loss can be different than the one used during training 
	
	norm = F.basic_math.absolute(Z)
	norm = F.basic_math.pow(norm,2.0)
	norm = F.sum(norm, axis=-1)
	norm = F.basic_math.pow(norm,1.0/2.0)
	return norm

def euclidian_joints(Z):

	norm = F.basic_math.absolute(Z)
	norm = F.basic_math.pow(norm,2.0)
	norm = F.sum(norm, axis=-1)
	norm = F.basic_math.pow(norm,1.0/2.0) 
	return norm.data

def max_euclidian_joints(Z):

	norm = F.basic_math.absolute(Z)
	norm = F.basic_math.pow(norm,2.0)
	norm = F.sum(norm, axis=-1)
	norm = F.basic_math.pow(norm,1.0/2.0) 
	norm = F.max(norm, axis =-1) 
	return norm.data

def number_frames_within_dist(Z, dist = 150):

	norm = F.basic_math.absolute(Z)
	norm = F.basic_math.pow(norm,2.0)
	norm = F.sum(norm, axis=-1)
	norm = F.basic_math.pow(norm,1.0/2.0) # shape[-1] is J
	norm = F.max(norm, axis =-1) 

	n_frames = xp.sum((norm.data <= dist), axis = -1)

	return n_frames


def compute_ProbLoss_alpha05(predictions, gt, score):

	n_predictions = predictions.shape[0]

	if n_predictions == 1:
		term2 = chainer.Variable(cuda.cupy.asarray(0.0).astype('float32'))
	else:
		d = predictions.shape[1]
		predictions_3d_matrix = xp.repeat(predictions,n_predictions,axis=0).astype(np.float32)
		predictions_3d_matrix = predictions_3d_matrix.reshape(n_predictions,n_predictions,d)
		Z = predictions_3d_matrix - predictions
		Z = chainer.Variable(Z)
		term2 = F.sum(score.bnorm(Z)) / (n_predictions * (n_predictions - 1))
	
	Zgt = predictions - gt
	Zgt = chainer.Variable(Zgt)
	term1 = F.sum(score.bnorm(Zgt)) / n_predictions
	obj = term1 - 0.5 * term2 
	return obj.data

def compute_SQRT_EuclidianError(predictions, gt):

	n_predictions = predictions.shape[0]
	Zgt = predictions - gt
	Zgt = chainer.Variable(Zgt)
	term1 = F.sum(l2norm(Zgt, 1.0)) / n_predictions
	return term1.data


def compute_MEU(predictions, J, score):

	# create matrix of losses
	n_predictions = predictions.shape[0]
	d = predictions.shape[1]
	
	predictions_3d_matrix = xp.repeat(predictions,n_predictions,axis=0).astype(np.float32)
	predictions_3d_matrix = predictions_3d_matrix.reshape(n_predictions,n_predictions,d)
	
	Z = predictions_3d_matrix - predictions
	Z_flatten = chainer.Variable(Z)
	Z = F.reshape(Z_flatten, (Z_flatten.data.shape[:-1] + (J, 3)))
	
	mean_per_joint_error_loss_matrix = xp.sum(euclidian_joints(Z), axis = -1) / J
	mean_error_loss_matrix = euclidian_joints(Z_flatten) / (3*J)
	max_error_loss_matrix = max_euclidian_joints(Z)
	training_scoring_matrix = score.bnorm(Z_flatten).data
	fraction_frame_loss_matrix = number_frames_within_dist(Z)

	MEU_vector_scoring = xp.sum(training_scoring_matrix, axis = 1)
	MEU_vector_me_per_joint = xp.sum(mean_per_joint_error_loss_matrix, axis = 1)
	MEU_vector_max = xp.sum(max_error_loss_matrix, axis = 1)
	MEU_vector_me = xp.sum(mean_error_loss_matrix, axis = 1)

	y_me_per_joint = xp.argmin(MEU_vector_me_per_joint)
	y_me = xp.argmin(MEU_vector_me)
	y_max = xp.argmin(MEU_vector_max)
	y_ff = xp.argmax(fraction_frame_loss_matrix)
	y_scoring = xp.argmin(MEU_vector_scoring)

	return y_me_per_joint, y_max, y_ff, y_scoring, y_me

def values_in_mm(evaluator):
	
	joints_me = xp.reshape(evaluator.poses_me, (evaluator.poses_me.shape[0], J,3))
	joints_max = xp.reshape(evaluator.poses_max, (evaluator.poses_max.shape[0],J,3))
	joints_ff = xp.reshape(evaluator.poses_ff, (evaluator.poses_ff.shape[0], J,3))
	
	#joints_me[:,:,2] = 0
	#joints_max[:,:,2] = 0
	#joints_ff[:,:,2] = 0
	
	gt = xp.asarray(evaluator.y).astype('float32')
	gt = xp.reshape(gt, (gt.shape[0], J, 3))
	#gt[:,:,2] =0
	#print gt.shape, joints_me.shape
	vector_me = chainer.Variable((joints_me - gt).astype('float32'))
	vector_max = chainer.Variable((joints_max - gt).astype('float32'))
	vector_ff = chainer.Variable((joints_ff - gt).astype('float32'))
	
	euclidian_distance = euclidian_joints(vector_me)
	max_error = max_euclidian_joints(vector_max)
	ff_error = number_frames_within_dist(vector_ff)

	return euclidian_distance, max_error, ff_error


class Evaluator():
	def __init__(self, model, X, y, score, loader):
		self.model = model
		self.x = X
		self.y = y
		self.score = score
		self.loader = loader
		self.predictions = []
		self.z_all = chainer.Variable(xp.asarray(np.random.uniform(-1, 1,(N_sampling,1, nrand)).astype(np.float32)))
	
	def generate_N_samples(self, N_sampling ):

		
		n = self.y.shape[0]
		d = self.y.shape[1]
		predictions = xp.zeros((n,N_sampling, d)).astype('float32')
		mean_predictions = xp.zeros((n, d)).astype('float32')

	#	print self.x.shape
		for i in range(n):
			
			x_i = self.x[i,:,:,:][xp.newaxis]

			# Generate N_sampling poses for a depth image
			predictions_i = xp.asarray(xp.zeros((N_sampling, d))).astype('float32')
	#		print x_i.ndim
			x=x_i
			#x = chainer.Variable(x_i.astype('float32')) 
			#print x.shape
			#print "x", x[:,:,20,20]
			image_features = self.model.fast_sample_depth_image(x)
			#print image_features.data
			image_features_all = chainer.Variable(xp.asarray(xp.tile(image_features.data, \
						(N_sampling,1)).astype('float32')))
			#print image_features_all.data
			z_all = chainer.Variable(xp.asarray(np.random.uniform(-1, 1,(N_sampling, self.model.nrand)).astype(np.float32)))
			predictions_i = self.model.fast_sample(z_all, image_features_all)

			#com_i = xp.asarray(seq.data[self.datas.indexes[i]].com)
			# reshape for denormalization
			predictions_i = xp.reshape(predictions_i.data,(N_sampling, J, 3))
	 
			predictions_i = xp.reshape(predictions_i, ((N_sampling, J * 3)))
			predictions_i *= max_y
			#predictions_i = self.loader.jointsImgTo3D(predictions_i)
			#print "pr2 ", predictions_i
			#print "pred", predictions_i
			
			#print "y", self.y[i]
			#torso = predictions_i[:, 6]
		#	print torso.shape, predictions_i[:,::3].shape
			predictions_i[:,::3] *= 640/128
			predictions_i[:,1::3] *= 480/128
			
			predictions_i = self.loader.jointsImgTo3D(predictions_i)
			
			#print "pr ", predictions_i[0,:]
			
			#for j in range(N_sampling):
			#	predictions_i[j,::3]  -= 150 - torso[j]
			#print predictions_i.shape
			#print predictions.shape
			predictions[i, :, :] = xp.asarray(predictions_i)
			mean_predictions[i,:] = xp.asarray(xp.mean(predictions_i, axis = 0))
		#predictions /= 1000
		#print predictions[:,1,1]
		self.predictions = chainer.Variable(predictions.astype('float32'))
		self.predicted_poses = predictions
		self.poses = mean_predictions
		return predictions.data



	def compute_Losses_samples(self):

		n = self.y.shape[0]
		d = self.y.shape[1]
		N_sampling = self.predicted_poses.shape[1]
		probloss = np.zeros(n)
		sqrt_eu_err = np.zeros(n)
		for i in range(n):
			y = xp.asarray(self.y[i,:][xp.newaxis]).astype('float32')
			probloss[i] = compute_ProbLoss_alpha05(self.predicted_poses[i,:,:], y, self.score)
			sqrt_eu_err[i] = compute_SQRT_EuclidianError(self.predicted_poses[i,:,:], y)

		return probloss, sqrt_eu_err

	def compute_Losses_PE(self):

		n = self.y.shape[0]
		d = self.y.shape[1]
		N_sampling = self.predicted_poses.shape[1]
		probloss = np.zeros(n)
		sqrt_eu_err = np.zeros(n)
		for i in range(n):
			y = xp.asarray(self.y[i,:][xp.newaxis]).astype('float32')
			probloss[i] = compute_ProbLoss_alpha05(self.poses_scoring[i,:][np.newaxis].astype('float32'), y, self.score)
			sqrt_eu_err[i] = compute_SQRT_EuclidianError(self.poses_sqrt_euc[i,:][np.newaxis].astype('float32'), y)

		return probloss, sqrt_eu_err
	


	def get_point_estimates(self):

		n = self.y.shape[0]
		d = self.y.shape[1]
		self.poses_me = xp.zeros((n,d))
		self.poses_max = xp.zeros((n,d))
		self.poses_ff = xp.zeros((n,d))
		self.poses_scoring = xp.zeros((n,d))
		self.poses_sqrt_euc = xp.zeros((n,d))
		for i in range(n):

			y = xp.asarray(self.y[i,:][xp.newaxis]).astype('float32')
			y_me, y_max, y_ff, y_scoring, y_sqrt_euc = compute_MEU(self.predicted_poses[i,:,:], J, self.score)
			self.poses_me[i,:] = self.predicted_poses[i,int(y_me),:]
			self.poses_max[i,:] = self.predicted_poses[i,int(y_max),:]
			self.poses_ff[i,:] = self.predicted_poses[i,int(y_ff),:]
			self.poses_scoring[i,:] = self.predicted_poses[i,int(y_scoring),:]
			self.poses_sqrt_euc[i,:] = self.predicted_poses[i,int(y_sqrt_euc),:]
	
	def get_point_estimates_3M(self, MEU = True, random = False):

		if not MEU:
			if not random:
				self.poses_me[:,:] = self.poses
				self.poses_max[:,:] = self.poses
				self.poses_ff[:,:] = self.poses
				self.poses_scoring[:,:] = self.poses
				self.poses_sqrt_euc[:,:] = self.poses
			else:
				n = self.predicted_poses.shape[0]
				n_predictions = self.predicted_poses.shape[1]
				for i in range(n):
					idx = np.random.randint(0,n_predictions)
					self.poses_me[i,:] = self.predicted_poses[i,idx]
					self.poses_max[i,:] = self.predicted_poses[i,idx]
					self.poses_ff[i,:] = self.predicted_poses[i,idx]
					self.poses_scoring[i,:] = self.predicted_poses[i,idx]
					self.poses_sqrt_euc[i,:] = self.predicted_poses[i,idx]
		else:
			self.get_point_estimates()

"""

"""
			


def evaluate(results_dir, beta, seed, alpha, C, cov, MEU, random,  nrand, X_test, Y_test, N_sampling, loader, J = 15):

	N = X_test.shape[0]
	distances = np.arange(0.0,81,5)
	line = np.zeros(len(distances))
	fraction_frame = 0.0
	probloss = np.zeros(N)
	sqrt_euc = np.zeros(N)
	probloss_PE = np.zeros(N)
	sqrt_euc_PE = np.zeros(N)
	euclidian_distance = np.zeros((N, J))
	max_euclidian_distance = np.zeros(N)

	batchsize = 128

	

	scoring = Score(beta, alpha, N_sampling)
	if mode == GAN:
		model = Generator(scoring)
		serializers.load_npz('HumanTryRelease/generator_human.model', model)
	else:
		model = JointPositionExtractor(scoring, nrand, J) 
		
		serializers.load_npz('HumanTryRelease/model_end.model', model)
	if use_gpu:
		model.to_gpu(gpu_id)
	
	
	with cupy.cuda.Device(gpu_id):
		start = 0
		end = 0
		batchsize = 100
		while True:
			start = end
			end = min([start + batchsize, N])
			indexes = np.arange(start,end)
		

			x_test = xp.asarray(X_test[indexes,:,:,:]).astype(xp.float32)
			y_test = xp.asarray(Y_test[indexes]).astype(xp.float32)

			#print x_test[1,:,10,10], x_test[15,:,10,10]
			#x_test = chainer.Variable(xp.asarray(x_test).astype(xp.float32))
			#y_test = chainer.Variable(xp.asarray(y_test).astype(xp.float32))
			
			#print("Generating samples")		
			evaluator = Evaluator( model, x_test, y_test, scoring, loader)					
			
			#x_test = chainer.Variable(xp.asarray(x_test).astype(xp.float32))
	
			evaluator.generate_N_samples(N_sampling) # Generate predictions, already denormed
			
			
			#print("Computing Probabilistic metrics")	
			l1, l2 = evaluator.compute_Losses_samples() 
			probloss[start:end] = l1 
			sqrt_euc[start:end] = l2 
			#print ("\n probloss :", l1, "\n sqrt_euc :" , l2)

			#print("Computing NON Probabilistic metrics")	
			evaluator.get_point_estimates_3M(MEU, random) # generate point estimates
			l1, l2 = evaluator.compute_Losses_PE() 
			probloss_PE[start:end] = l1 
			sqrt_euc_PE[start:end] = l2 
			#print ("\n probloss_PE: ", l1, "\n sqrt_eucl_PE: ", l2)
				
			#print("Computing Errors")	
			m1,m2,m3 = values_in_mm(evaluator)
			m1 = cuda.to_cpu(m1)
			m2 = cuda.to_cpu(m2)
			euclidian_distance[start:end, :] = m1
			max_euclidian_distance[start:end] = m2 
			fraction_frame += m3 / float(N) 
			#print ("\n euclidian distance:", m1, "\n max_euclidian_distance", m2)
			#if (alpha == 0.0):
			#	print("Computing FF curves")	
		#		line_bi = compute_ff_line_PE(evaluator_PE.predicted_poses, sub_gt3D_array, 14, distances)
			
		#		line += line_bi / float(N)
			
			if end == N:
				break
		#the lines below were added by me
		MeJEE = xp.mean(euclidian_distance)
		MaJEE = xp.mean(max_euclidian_distance)
		ProbLoss = xp.mean(probloss)
		ProbLoss_PE = xp.mean(probloss_PE)
		print "Probloss: ", ProbLoss , "\n MeJEE: ", MeJEE, "\n MaJEE: ", MaJEE
		print ("\n Probloss PE: ", ProbLoss_PE)
		print ("\n FF: ", fraction_frame)
		#if alpha == 0:
	#		print("\n FF_line: ", line)"""


if __name__ == '__main__':

	global gpu_id
	use_gpu = True
	gpu_id = 3
	use_gpu = 1
	if use_gpu:
		xp = cuda.cupy
	else:
		xp = np
	J = 15
	loader = dataloader('data/Subject1_rgbd_images/')
	loader.load_images()

	loader.load_joints()
	loader.uvd_joints = loader.joints_to_uvd()
	loader.cut()
	X_test = np.asarray(loader.images)
	X_test /= np.max(X_test)
	X_test = np.reshape(X_test, (X_test.shape[0],1, X_test.shape[1], X_test.shape[2]))
	Y_test = xp.asarray(loader.uvd_joints)
	max_y = np.max(Y_test)
	#Y_test = xp.asarray(loader.joints)
	#Y_test /= np.max(Y_test)
	Y_test[:,::3] *= 640/128
	Y_test[:,1::3] *= 480/128
	Y_test = loader.jointsImgTo3D(Y_test)
	#Y_test /= 1000

	
	tmp1, x_test, tmp2, y_test = train_test_split(X_test, Y_test, random_state = 42) 
	X_test, Y_test = x_test, y_test
	print X_test.shape, Y_test.shape
	print np.allclose(X_test[1,:,:,:], X_test[20,:,:,:])
	N = X_test.shape[0]#	
	np.random.seed(0)
	beta = 1.0
	seed = 0
	alpha =0#0.5

	C = 1e-3
	nrand = 0 #200
	results_dir = 'HumanTryRelease/'
	cov = 5
	MEU = True
	N_sampling =100
	random = False
	evaluate(results_dir, beta, seed, alpha, C, cov, MEU, random,  nrand,  X_test, Y_test, N_sampling, loader, J = 15)
	#print ("\n Testing with alpha = 0")
	#alpha = 0.0
	#evaluate(results_dir, beta, seed, alpha, C, cov, MEU, random,  nrand, testSeqs, X_test, Y_test, N_sampling, J = 14)
	
