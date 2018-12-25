import numpy as np

class nn:

	def __init__(self, input_layer):
		self.layers = [input_layer]
		self.nlayers = 1;

	def add(self,layer):
		self.layers.append(layer)
		self.nlayers += 1

	def forward(self,input_data):
		

	def showArchitecture(self):
		print("##############################################")
		for i in range(self.nlayers):
			print("Layer:",self.layers[i].getName()," -",
				           self.layers[i].getNumNodes(),"nodes ("+
				           self.layers[i].getActivation()+")")
		print("##############################################")


class layer:

	def __init__(self,n_nodes,activation,layer_name='',weights=None):
		self.name = layer_name
		self.n_nodes = n_nodes
		self.activation = activation
		self.input = []
		if (weights == None):
			self.weights = np.random.randn(1,n_nodes+1) #bias
		else:
			self.weights = weights

	def showFullDetails(self):
		print("#######################")
		print("Layer:",self.name)
		print("Activation function:",self.activation)
		print("Weigths:")
		print(self.weights)
		print("#######################")

	def getName(self):
		return self.name

	def getNumNodes(self):
		return self.n_nodes

	def getActivation(self):
		return self.activation

	def setInputData(self, input_data):
		self.input = input_data