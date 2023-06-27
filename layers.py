"""FHWS/MAI/ANN_SS22 Assignment 1 - feed forward network layers

Created: Magda Gregorova, 20/4/2022
"""

import torch

class Linear:
	"""Apply linear transformation to input data y = X W^T + b.

	Attributes:
		W: torch.tensor of shape (out_feartures, in_features) - weight matrix
		b: torch.tensor of shape (1, out_features) - bias vector
		ins: torch.tensor of shape (num_instances, in_features) - input data
		outs: torch.tensor of shape (n, out_features) - output data
		W.g: torch.tensor of shape (out_feartures, in_features) - weight matrix global gradients
		b.g: torch.tensor of shape (1, out_features) - bias vector global gradients
		ins.g: torch.tensor of shape (num_instances, in_features) - input data global gradients
	"""

	def __init__(self, W, b):
		"""Initiate instances with weight and bias attributes.

		Arguments:
			W: torch.tensor of shape (out_feartures, in_features) - weight matrix
			b: torch.tensor of shape (1, out_features) - bias vector
		""" 
		self.W=W
		self.b=b
		
		
		
		
	def forward(self, ins):
		"""Forward pass through linear transformation. Populates ins and outs attributes.

		Arguments:
			ins: torch.tensor of shape (num_instances, in_features) - input data

		Returns:
			torch.tensor of shape (num_instances, out_features) - output data
		""" 
		self.ins=ins
		
		self.outs=torch.matmul(self.ins,self.W.T)
		
		self.outs+=self.b
		
		
		
		return self.outs

	def backward(self, gout):
		"""Backward pass through linear transformation. Populates W.g, b.g and ins.g attributes.

		Arguments:
			gout: torch.tensor of shape (num_instances, out_features) - upstream gradient

		Returns:
			torch.tensor of shape (num_instances, num_dims) - input data global gradients
		""" 
		self.gout=gout

		self.W.g=self.gout.T @ self.ins
		self.b.g=torch.ones(1,self.gout.shape[0])@self.gout
		self.ins.g=self.gout @ self.W


		return self.ins.g


class Relu:
	"""Apply relu non-linearity x = max(0, x).

	Attributes:
		ins: torch.tensor of shape (num_instances, num_dims) - input data
		outs: torch.tensor of shape (num_instances, num_dims) - output data
		ins.g: torch.tensor of shape (num_instances, num_dims) - input data global gradients

	"""

	def forward(self, ins):
		"""Forward pass through relu. Populates ins and outs attributes.

		Arguments:
			
            ins: torch.tensor of shape (num_instances, num_dims) - input data

		Returns:
			torch.tensor of shape (num_instances, num_dims) - output data
		""" 
		self.ins=ins
        
		zero=torch.zeros_like(ins)
		self.outs=torch.max(zero,self.ins)
		
		"""   
		self.outs=torch.zeros_like(ins)
		for i in range(ins.shape[0]):
			for j in range(ins.shape[1]):
				if ins[i][j]>0:
					self.outs[i][j]=ins[i][j]
				else:
					self.outs[i][j]=0"""

		return self.outs

	def backward(self, gout):
		"""Backward pass through relu. Populates ins.g attributes.

		Arguments:
			gout: torch.tensor of shape (num_instances, num_dims) - upstream gradient

		Returns:
			torch.tensor of shape (num_instances, num_dims) - input data global gradients
		""" 
		self.gout=gout
		self.ins.g=torch.zeros_like(self.gout)
		for i in range(self.ins.shape[0]):
			for j in range(self.ins.shape[1]):           
				if self.ins.g[i,j]<=0:
					self.ins.g[i,j]=0*self.gout[i,j]                   
				if self.ins[i,j]>0:
					self.ins.g[i,j]=1*self.gout[i,j]
		return self.ins.g


class Model():
	"""Neural network model.

	Attributes:
		layers: list of NN layers in the order of the forward pass from inputs to outputs
	"""

	def __init__(self, layers):
		"""Initiate model instance all layers. 

		Layers are expected to be instances of Linear and Relu classes.
		The shall be passed to Model instances as a list in the correct order of forward execution.

		Arguments:
			layers: list of layer instances		
		"""
		self.layers = layers

	def forward(self, ins):
		"""Forward pass through model. 

		Arguments:
			ins: torch.tensor of shape (num_instances, in_features) - input data

		Returns:
			torch.tensor of shape (n, out_features) - model predictions
		""" 
		outs = ins
		#print("OUTS   ==  ", outs)   
		for layer in self.layers:          
			outs = layer.forward(outs)          
			#print("OUTS   ==  ", outs)
            
		return outs

	def backward(self, gout):
		"""Backward pass through model

		Arguments:
			gout: torch.tensor of shape (num_instances, out_features) - gradient of loss with respect to predictions

		Returns:
			torch.tensor of shape (n, in_features) - gradient with respect to forward inputs
		""" 
		for layer in reversed(self.layers):
			gout = layer.backward(gout)
		return gout

