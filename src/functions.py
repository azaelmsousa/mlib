import numpy as np
import math

possibleActivations = ['ReLU','sigmoid','tanH']

def ReLU(x):
	if (x > 0):
		return x
	else:
		return 0

def dReLU(x):
	if (x > 0):
		return 1
	else:
		return 0

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def dsigmoid(x):
	sig = sigmoid(x)
	return  sig * (1 - sig)

def tanH(x):
	return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def dtanH(x):
	return  4*np.exp(2*x) / ((np.exp(2*x)+1)*(np.exp(2*x)+1))
