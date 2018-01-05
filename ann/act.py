import numpy as np
import scipy.special


def linear(z):
	return z


def d_linear(a):
	return 1


def relu(z):
	return z * (z > 0)


def d_relu(a):
	return a > 0


def tanh(z):
	return np.tanh(z)


def d_tanh(a):
	return 1 - a ** 2


def sigmoid(z):
	return scipy.special.expit(z)


def d_sigmoid(a):
	return a * (1 - a)


def sigmoid_with_binary_xentropy(z):
	"""To be used in conjunction with loss.binary_xentropy_with_sigmoid"""
	return sigmoid(z)


def softmax(z):
	exps = np.exp(z - np.max(z, axis=1, keepdims=True))
	return exps / np.sum(exps, axis=1, keepdims=True)


def softmax_with_xentropy(z):
	"""To be used in conjunction with loss.xentropy_with_softmax"""
	return softmax(z)


def get_d_act(act):
	if act == linear:
		return d_linear
	elif act == relu:
		return d_relu
	elif act == tanh:
		return d_tanh
	elif act == sigmoid:
		return d_sigmoid
	elif act == softmax:
		# derivative depends on derivative of loss function and thus will be calculated in layer backward method
		return None
	elif act == softmax_with_xentropy or act == sigmoid_with_binary_xentropy:
		# derivative can be simplified and thus will be calculated in layer backward method
		return None
	else:
		raise NotImplementedError("No derivative for activation function '{}'".format(act.__name__))
