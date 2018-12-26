import numpy as np
import functions as f

class nn:

	def __init__(self, input_layer):
		self.layers = [input_layer]
		self.nlayers = 1;

	def add(self,layer):
		if (self.layers[-1].getNumNodes() != layer.getNodesPreviousLayer()):
			print("Error on class nn, method add: Number of nodes of current layer must be equal to the previous one")
			exit()
		self.layers.append(layer)
		self.nlayers += 1

	def forward(self,x):
		x_ = np.copy(x)
		for l in self.layers:
			for n in l.nodes:
				w = n.getWeights()
				s = np.matmul(x_,w)
				if (l.getActivation() in f.possibleActivations):
					a = getattr(f,l.getActivation())(s)
				else:
					a = s
				#join outputs of each node
		return 

	def showArchitecture(self):
		print("##############################################")
		for i in range(self.nlayers):
			print("Layer:",self.layers[i].getName()," -",
				           self.layers[i].getNumNodes(),"nodes ("+
				           self.layers[i].getActivation()+")")
		print("##############################################")


class layer:

	def __init__(self,n_nodes,n_nodes_previous_layer,activation,layer_name='',weights=None):
		self.name = layer_name
		self.n_nodes = n_nodes
		self.activation = activation
		self.n_nodes_previous_layer = n_nodes_previous_layer
		self.nodes = []
		self.input = []

		for i in range(n_nodes):
			self.nodes.append(node(n_nodes_previous_layer+1,weights))

	def showFullDetails(self):
		print("#######################")
		print("Layer:",self.name)
		print("Activation function:",self.activation)
		print("#######################")

	def getName(self):
		return self.name

	def getNumNodes(self):
		return self.n_nodes

	def getActivation(self):
		return self.activation

	def getNodesPreviousLayer(self):
		return self.n_nodes_previous_layer

	def setInputData(self, input_data):
		self.input = input_data

class node:

	def __init__(self,n_weights,weights=None):
		if (weights == None):
			self.weights = np.random.randn(1,n_weights+1) #bias
		else:
			self.weights = weights

	def getWeights(self):
		return self.weights

