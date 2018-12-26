import numpy as np
from src import functions as f

class nn:

	def __init__(self):
		self.layers = []
		self.n_layers = 0

	def add(self,layer):
		if (self.n_layers > 0):
			if (self.layers[-1].getNumNodes() != layer.getNodesPreviousLayer()):
				print("Error on class nn, method add: Number of nodes of current layer must be equal to the previous one")
				exit()
		self.layers.append(layer)
		self.n_layers += 1

	def forward(self,x):
		x_ = np.copy(x)
		for l in self.layers:
			x_aux = np.array([[]])
			for n in l.nodes:
				w = n.getWeights()
				s = np.matmul(x_,w)
				if (l.getActivation() in f.possibleActivations):
					a = getattr(f,l.getActivation())(s)
				else:
					a = s
				x_aux = np.append(x_aux,[a],0)
			x_ = np.copy(x_aux)
		return 

	def showArchitecture(self):
		print("##############################################")
		for i in range(self.n_layers):
			print("Layer:",self.layers[i].getName()," -",
				           self.layers[i].getNodesPreviousLayer(),"->",
				           self.layers[i].getNumNodes(),"nodes ("+
				           self.layers[i].getActivation()+")")
			print("Nodes:")
			for j in range(self.layers[i].getNumNodes()):
				print(self.layers[i].nodes[j].getWeights())
			print("\n")
		print("##############################################")


class layer:

	def __init__(self,n_nodes,n_nodes_previous_layer,activation,layer_name='',weights=None):
		self.name = layer_name
		self.n_nodes = n_nodes
		self.activation = activation
		self.n_nodes_previous_layer = n_nodes_previous_layer
		self.nodes = []
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

	def getNodes(self):
		return self.nodes

class node:

	def __init__(self,n_weights,weights=None):
		if (weights == None):
			self.weights = np.random.randn(1,n_weights+1) #bias
		else:
			self.weights = weights

	def getWeights(self):
		return self.weights

