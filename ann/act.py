import numpy as np
import scipy.special


def linear(z):
	return z


def d_linear(a):
	return 1


def relu(z):
	return z * (z > 0)


def d_relu(a):
	return np.int64(a > 0)


def tanh(z):
	return np.tanh(z)


def d_tanh(a):
	return 1 - a ** 2


def sigmoid(z):
	return scipy.special.expit(z)


def d_sigmoid(a):
	return a * (1 - a)


def softmax(z):
	exps = np.exp(z - np.max(z, axis=1, keepdims=True))
	return exps / np.sum(exps, axis=1, keepdims=True)


def softmax_with_cross_entropy(z):
	# separate function only for later reference
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
		# derivative depends on activation function and will be calculated in backward method in layer
		return None
	elif act == softmax_with_cross_entropy:
		# derivative depends on activation function and will be calculated in backward method in layer
		return None
	else:
		raise NotImplementedError("No derivative for activation function '{}'".format(act.__name__))
