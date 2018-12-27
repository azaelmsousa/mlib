import numpy as np
from src import functions as f

np.random.seed(42)

class nn:

	def __init__(self):
		self.layers = []
		self.n_layers = 0
		self.input_data = None

	def add(self,layer):
		if (self.n_layers > 0):
			if (self.layers[-1].getNumNodes() != layer.getNodesPreviousLayer()):
				print("Error on class nn, method add: Number of nodes of current layer must be equal to the previous one")
				exit()
		self.layers.append(layer)
		self.n_layers += 1

	def loadInput(self, input_data):
		aux = np.insert(input_data,0,1,axis=1)
		self.input_data = aux

	def forwardSamples(self,x):
		x_ = np.insert(x,0,1)
		for l in self.layers:
			x_s = []
			nodes = l.getNodes()
			for n in nodes:
				w = n.getWeights()
				s = np.matmul(x_,w.transpose())	
				x_s.append(s)
			x_s = np.array(x_s)
			if (l.getActivation() in f.possibleActivations):
				a = getattr(f,l.getActivation())(x_s)
			else:
				print("Warning, the activation function",l.getActivation(),"for layer",l.getName(),
					  " is not valid. No function used.")
				a = x_s
			a = np.insert(a,0,1) #inserting bias
			x_ = np.copy(a)
		output = np.delete(x_,0) #removing bias
		return output


	def forward(self):
		x_ = np.copy(self.input_data)
		for l in self.layers:
			x_s = np.array([[]])
			nodes = l.getNodes()
			for n in nodes:
				w = n.getWeights()
				s = np.matmul(x_,w.transpose())	
				x_s = s if x_s.size==0 else np.concatenate((x_s,s),axis=1)			
			if (l.getActivation() in f.possibleActivations):
				a = getattr(f,l.getActivation())(x_s)
			else:
				print("Warning, the activation function",l.getActivation(),"for layer",l.getName(),
					  " is not valid. No function used.")
				a = x_s
			a = np.insert(a,0,1,axis=1) #inserting bias
			x_ = np.copy(a)
		output = np.delete(x_,0,axis=1) #removing bias
		return output

	def predict(self):
		output = self.forward()
		print(np.argmax(output,axis=1))

	def showArchitecture(self):
		print("##############################################")
		for i in range(self.n_layers):
			print("Layer:",self.layers[i].getName()," -",
				           self.layers[i].getNodesPreviousLayer(),"->",
				           self.layers[i].getNumNodes()," ("+
				           self.layers[i].getActivation()+") weights shape:",
				           self.layers[i].getNodesPreviousLayer()+1)
			print("--- Input data shape:",self.input_data.shape)
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
			if (weights is None):
				self.nodes.append(node(n_nodes_previous_layer+1,None))
			else:
				if (len(weights) != n_nodes):
					self.nodes.append(node(n_nodes_previous_layer+1,None))
				else: 	
					self.nodes.append(node(n_nodes_previous_layer+1,weights[i]))


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

	def getSingleNode(self,i):
		if ((i < 0) or (i >= len(self.nodes))):
			print("Error on class layer, method getSingleNode: Index out of bounderies.")
			exit()
		return self.nodes[i]

class node:

	def __init__(self,n_weights,weights=None):
		if (weights is None):
			self.weights = np.random.uniform(0,1,(1,n_weights))
		else:
			self.weights = weights

	def getWeights(self):
		return self.weights

