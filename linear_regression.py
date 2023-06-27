"""FHWS/MAI/ANN_SS22 Assignment 1 - linear regression

Created: Magda Gregorova, 17/4/2022
"""
import torch

def linear_single_forward(x, w, b):
	"""Linear model for single input - forward pass (naive implementation with for loops).

	Arguments:
	x: torch.tensor of shape (d) - input instance
	w: torch.tensor of shape (d) - weight vector
	b: float - bias

	Returns:
	out: torch.tensor of shape (1) - output of linear transform
	cache: tuple (x, w, b)
	"""

	# forward pass - compute predictions iteratively
	num_dims = x.shape[0]

	out = torch.zeros(1)
	for i in range(num_dims):
		out += x[i] * w[i]
	out += b

	cache = (x, w, b)
	return out, cache


def squared_error_forward(yhat, y):
	"""Squared error loss - forward pass.

	Arguments:
	yhat: torch tensor of shape (1) - prediction
	y: torch tensor of shape (1) - true label

	Returns:
	loss: torch.tensor of shape (1) - squared error loss
	cache: tuple (yhat, y)
	"""

	# forward pass
	loss = (yhat - y)**2

	cache = (yhat, y)
	return loss, cache

def linear_single_lgrad(cache):
	"""Linear model for single input - local gradient (naive implementation with for loops).

	Arguments:
	cache: tuple (x, w, b)
		x: torch.tensor of shape (d) - input instance
		w: torch.tensor of shape (d) - weight vector
		b: float containing bias

	Returns:
	xg: torch.tensor of shape (d) - local gradient with respect to input
	wg: torch.tensor of shape (d) - local gradient with respect to input weight vector
	bg: float - local gradient with respect to bias
	"""

	x, w, b = cache
	xg = torch.zeros_like(x)
	wg = torch.zeros_like(w)
	bg = torch.zeros_like(b)

	x.requires_grad=True
	w.requires_grad=True
	b.requires_grad=True
	num_dims=x.shape[0]
	f=torch.zeros(1)
	for i in range(num_dims):
		f+=x[i]*w[i]
	f+=b[0]        
	f.backward()
	xg=x.grad
	wg=w.grad
	bg=b.grad
	return xg, wg, bg

def squared_error_lgrad(cache):
	"""Squared error loss - local gradient.

	Arguments:
	cache: tuple (yaht, y)
		yhat: torch tensor of shape (1) - prediction
		y: torch tensor of shape (1) - true label

	Returns:
	yhatg: torch tensor of shape (1) - local gradient with respect to yhat
	yg: torch tensor of shape (1) - local gradient with respect to y
	"""

	yhat, y = cache
	y.requires_grad=True
	yhat.requires_grad=True
	f=(yhat - y)**2
	f.backward()
	yhatg=yhat.grad
	yg=y.grad

	return yhatg, yg


def linear_single_ggrad(cache, gout):
	"""Linear model for single input - global gradient.

	Arguments:
	cache: tuple (xg, wg, bg)
		xg: torch.tensor of shape (d) - local gradient with respect to input
		wg: torch.tensor of shape (d) - local gradient with respect to input weight vector
		bg: float - local gradient with respect to bias
	gout: torch.tensor of shape (1) - upstream global gradient

	Returns:
	xgrad: torch.tensor of shape (d) - global gradient with respect to input
	wgrad: torch.tensor of shape (d) - global gradient with respect to input weight vector
	bgrad: float - global gradient with respect to bias
	"""

	xg, wg, bg = cache
	xgrad=xg*gout
	wgrad=wg*gout
	bgrad=bg*gout
    
	return xgrad, wgrad, bgrad


def linear_forward(X, w, b):
	"""Linear model - forward pass.

	Arguments:
	X: torch.tensor of shape (n, d) - input instances
	w: torch.tensor of shape (d, 1) - weight vector
	b: float - bias

	Returns:
	out: torch.tensor of shape (n, 1) - outputs of linear transform
	cache: tuple (X, w, b)
	"""

	out=torch.mm(X,w)+b
	 
	cache=(X,w,b)

	return out, cache


def mse_forward(yhat, y):
	"""MSE loss - forward pass.

	Arguments:
	yhat: torch tensor of shape (n, 1) - prediction
	y: torch tensor of shape (n, 1) - true label

	Returns:
	loss: torch.tensor of shape (1) - squared error loss
	cache: tuple (yhat, y)
	"""
	loss=(yhat-y)**2
	loss=loss.sum() * (loss.shape[0])**-1
	cache=(yhat,y)
	return loss, cache


def linear_backward(cache, gout):
	"""Linear model - backward pass.

	Arguments:
	cache: tuple (X, w, b)
		X: torch.tensor of shape (n, d) - input instances
		w: torch.tensor of shape (d, 1) - weight vector
		b: float - bias
	gout: torch.tensor of shape (n, 1) - upstream global gradient

	Returns:
	xgrad: torch.tensor of shape (n, d) - global gradient with respect to input
	wgrad: torch.tensor of shape (d, 1) - global gradient with respect to input weight vector
	bgrad: float - global gradient with respect to bias
	"""
	X,w,b=cache
	Xgrad=torch.matmul(gout,w.T)
	wgrad=torch.matmul(X.T,gout)
	bgrad=gout.sum()
	#print(f"xg={Xgrad},wg={wgrad},bg={bgrad},gout={gout}")
	return Xgrad, wgrad, bgrad

def mse_backward(cache):
	"""MSE loss - backward pass.
	Arguments:
	cache: tuple (yaht, y)
		yhat: torch tensor of shape (n, 1) - prediction
		y: torch tensor of shape (n, 1) - true label
	Returns:
	yhatgrad: torch tensor of shape (n, 1) - global gradient with respect to yhat
	ygrad: torch tensor of shape (n, 1) - global gradient with respect to y
	"""
	yhat,y=cache
	n=yhat.shape[0]    
	yhatgrad=(2*(yhat-y))/n
	ygrad=(-2*(yhat-y))/n

	return yhatgrad, ygrad

