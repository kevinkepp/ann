import numpy as np


def mse(a, y):
	m = y.shape[0]
	return np.sum(np.power(a - y, 2)) / m


def d_mse(a, y):
	return 2 * (a - y)


def xentropy(a, y):
	"""
	:param a: shape (m,n) or (m,) and values in [0,1]
	:param y: shape (m,n) or (m,) and values in {0,1}
	:return:
	"""
	m = y.shape[0]
	if y.ndim == 1 or y.shape[1] == 1:
		return binary_xentropy(a, y)
	a_y = np.sum(np.multiply(a, y), axis=1)  # class probabilities for correct classes
	nll = -np.log(a_y)
	return np.sum(nll) / m


def d_xentropy(a, y):
	print(
		"Warning: Cross entropy can be numerically unstable. Use loss.xentropy_with_softmax or "
		"loss.binary_xentropy_with_sigmoid when possible.")
	if y.ndim == 1 or y.shape[1] == 1:
		return d_binary_xentropy(a, y)
	return -np.divide(y, a)


def binary_xentropy(a, y):
	"""
	:param a: shape (m,1) or (m,) with values in [0,1]
	:param y: shape (m,1) or (m,) with values in {0,1}
	:return:
	"""
	m = y.shape[0]
	a_y = np.multiply(a, y) + np.multiply(1 - a, 1 - y)  # class probabilities for correct classes
	nll = -np.log(a_y)
	return np.sum(nll) / m


def d_binary_xentropy(a, y):
	print(
		"Warning: Cross entropy can be numerically unstable. Use loss.xentropy_with_softmax or "
		"loss.binary_xentropy_with_sigmoid when possible.")
	return np.divide(1 - y, 1 - a) - np.divide(y, a)


def xentropy_with_softmax(a, y):
	"""To be used in conjunction with activation function act.softmax_with_xentropy"""
	return xentropy(a, y)


def d_xentropy_with_softmax(a, y):
	# computes dz directly instead of computing first da and then dz
	return a - y


def binary_xentropy_with_sigmoid(a, y):
	"""To be used in conjunction with activation function act.sigmoid_with_xentropy"""
	return binary_xentropy(a, y)


def d_binary_xentropy_with_sigmoid(a, y):
	# computes dz directly instead of computing first da and then dz
	return a - y


def get_d_loss(loss):
	if loss == mse:
		return d_mse
	elif loss == xentropy:
		return d_xentropy
	elif loss == xentropy_with_softmax:
		return d_xentropy_with_softmax
	elif loss == binary_xentropy_with_sigmoid:
		return d_binary_xentropy_with_sigmoid
	else:
		raise NotImplementedError("No derivative for loss function '{}'".format(loss.__name__))
