import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.cross_validation import train_test_split
import numpy as np
import time
import sys
###################### ADD PATH TO THE DEEP PRIOR PACKAGE HERE
sys.path.append('../../DeepPrior/src/')
sys.path.append('../../utils/')
from data.dataset import NYUDataset
from data.importers import NYUImporter
import math

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
nrand = 0
beta = 1.0
alpha = 0.5

use_gpu = 0

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(size_out_1*size_out_2*n_filters + nrand, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 3*J)
        
        self.conv1 = nn.Conv2d(1, n_filters, n_conv)
        self.conv2 = nn.Conv2d(n_filters,n_filters,n_conv)
        self.conv3 = nn.Conv2d(n_filters, n_filters, n_conv)
        
        self.mp1 = nn.MaxPool2d(n_pool)
        self.mp2 = nn.MaxPool2d(n_pool)
        
    def forward(self, x):
		h = F.relu(self.conv1(x))
		h = self.mp1(h)
		h = F.relu(self.conv2(h))
		h = self.mp2(h)
		h = F.relu(self.conv2(h))
		print type(h)
		data = h.data
		        
	        data.resize_(x.size(0), size_out_1*size_out_2*n_filters)
		h = Variable(data)
		h = F.relu(self.fc1(h))
		h = F.relu(self.fc2(h))
		h = F.relu(self.fc3(h))
		
	        return h

class Discriminator(nn.Module):
		def __init__(self):
			super(Discriminator, self).__init__()
			self.conv1 = nn.Conv2d(1, n_filters, n_conv)
			self.conv2 = nn.Conv2d(n_filters,n_filters,n_conv)
			self.conv3 = nn.Conv2d(n_filters, n_filters, n_conv)
			
		        self.mp1 = nn.MaxPool2d(n_pool)
		        self.mp2 = nn.MaxPool2d(n_pool)

			self.fc1 = nn.Linear(size_out_1*size_out_2*n_filters + nrand + 3 * J, 200)
			self.fc2 = nn.Linear(200, 200)
			self.fc3 = nn.Linear(200, 1)
		def forward(self, x, z):
			h = F.relu(self.conv1(x))
			h = self.mp1(h)
			h = F.relu(self.conv2(h))
			h = self.mp2(h)
			h = F.relu(self.conv2(h))
			print type(h)



			data = h.data
			data.resize_(h.size(0), size_out_1*size_out_2*n_filters)
			h = Variable(data)
			print z.size(), h.size()
			h_extended = torch.cat([h, z],1)
			
			h = F.relu(self.fc1(h_extended))
			h = F.relu(self.fc2(h))
			h = F.sigmoid(self.fc3(h))
			return h
if __name__ == '__main__':
	print "start"
	generator = Generator()
	discriminator = Discriminator()
	print "cuda"
	discriminator.cuda()
	generator.cuda()
	print "opt"
	criterion = nn.BCELoss()
	d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0005)
	g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0005)
	
	#loading data
	di = NYUImporter('../../DeepPrior/data/NYU')
	Seq = di.loadSequence('test')
	trainDataset = NYUDataset([Seq])
	X_train, Y_train = trainDataset.imgStackDepthOnly('test')
	Y_train = np.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1]* Y_train.shape[2]))
	x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.2)
	N = y_train.shape[0]
	for epoch in range(Nepoch):
		sum_dis_loss = np.float32(0)
		sum_gen_loss = np.float32(0)
		#xp.random.shuffle(train_data)
		for i in range(0, N, batchsize):
			input_images = torch.FloatTensor(x_train[i:i+batchsize])
			images = Variable(input_images.cuda())
			real_poses = Variable(torch.FloatTensor(y_train[i:i+batchsize])).cuda()
			
			real_labels = Variable(torch.ones(images.size(0))).cuda()
			fake_labels = Variable(torch.zeros(images.size(0))).cuda()
		#train discriminator
			
			discriminator.zero_grad()
			outputs = discriminator(images, real_poses)
			real_loss = criterion(outputs, real_labels)
			real_score = outputs
			print "real loss" , real_loss			
			noise = Variable(torch.randn(images.size(0), 1,  images.size(2), images.size(3))).cuda()
			print images.size()
			fake_poses = generator(noise)
			outputs = discriminator(images,fake_poses)
			fake_loss = criterion(outputs, fake_labels)
			fake_score = outputs
			
			d_loss = real_loss + fake_loss
			d_loss.backward()
			d_optimizer.step()
			print "fake_loss", fake_loss
			# Train the generator 
			generator.zero_grad()
			noise = Variable(torch.randn(images.size(0), 1, images.size(2), images.size(3))).cuda()
			fake_poses = generator(noise)
			outputs = discriminator(images, fake_poses)
			g_loss = criterion(outputs, real_labels)
			g_loss.backward()
			g_optimizer.step()
			
			print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, ' 
			'D(x): %.2f, D(G(z)): %.2f' %(epoch, 200, i+1, 600, d_loss.data[0], 
			g_loss.data[0],real_score.data.mean(), fake_score.cpu().data.mean()))
