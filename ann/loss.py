import numpy as np


def mse(y_pred, y):
	m = y.shape[0]
	return np.sum(np.power(y_pred - y, 2)) / m


def d_mse(y_pred, y):
	return 2 * (y_pred - y)


def cross_entropy_binary(y_pred, y):
	"""y_pred - shape (m,1) and values in [0,1]"""
	m = y.shape[0]
	nll = -np.dot(y.T, np.log(y_pred)) - np.dot(1 - y.T, np.log(1 - y_pred))
	return np.sum(nll) / m


def d_cross_entropy_binary(y_pred, y):
	return np.divide(1 - y, 1 - y_pred) - np.divide(y, y_pred)


def cross_entropy(y_pred, y):
	"""y_pred - shape (m,n) and values in [0,1]"""
	m = y.shape[0]
	nll = -np.log(np.sum(np.multiply(y_pred, y), axis=1))
	return np.sum(nll) / m


def d_cross_entropy(y_pred, y):
	# FIXME this can be numerically unstable, is there a way to make it stable?
	return -np.divide(y, y_pred)


def cross_entropy_with_softmax(y_pred, y):
	"""to be used in conjunction with activation function act.softmax_with_cross_entropy"""
	return cross_entropy(y_pred, y)


def d_cross_entropy_with_softmax(y_pred, y):
	# calculates dz directly using LogSumExp trick
	return y_pred - y


def get_d_loss(loss):
	if loss == mse:
		return d_mse
	elif loss == cross_entropy_binary:
		return d_cross_entropy_binary
	elif loss == cross_entropy:
		return d_cross_entropy
	elif loss == cross_entropy_with_softmax:
		return d_cross_entropy_with_softmax
	else:
		raise NotImplementedError("No derivative for loss function '{}'".format(loss.__name__))
