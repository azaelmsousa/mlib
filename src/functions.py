import numpy as np
import math

possibleActivations = ['ReLU','sigmoid','tanh','softmax']
possibleLoss = ['crossEntropy']

#--------------------------------------------------------
#                   Activation Functions
#--------------------------------------------------------

def ReLU(x):
	y = np.zeros(x.shape)
	return np.where(x>0,x,y)

def dReLU(x):
	y = np.zeros(x.shape)
	z = np.ones(x.shape)
	return np.where(x>0,z,y)	

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def dsigmoid(x):
	sig = sigmoid(x)
	return  sig * (1 - sig)

def tanh(x):
	return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def dtanh(x):
	return 4*np.exp(2*x) / ((np.exp(2*x)+1)*(np.exp(2*x)+1))

def softmax(x):
	x = np.exp(x,dtype=np.float128)
	out = x / np.sum(x,axis=1,keepdims=True)
	return out

#--------------------------------------------------------
#                     Loss Functions
#--------------------------------------------------------

def crossEntropy(y_pred,y):
	# note that y must be the one hot enconding of the original class
	ln_y = np.log(y_pred)
	loss = np.matmul(ln_y,y)
	return loss





