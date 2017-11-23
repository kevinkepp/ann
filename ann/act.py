import numpy as np


def relu(v):
	# return np.maximum(0, v)
	return v * (v > 0)


def d_relu(v_relu):
	c = v_relu.copy()
	c[c > 0] = 1
	return c


def d_tanh(v_tanh):
	return 1 - v_tanh ** 2


def sigmoid(v):
	return 1 / (1 + np.exp(-v))


def d_sigmoid(v_sigmoid):
	return v_sigmoid * (1 - v_sigmoid)


def softmax(v):
	# unstable
	# exps = np.exp(v); return exps / np.sum(exps)
	# stable
	exps = np.exp(v - np.max(v))
	return exps / np.sum(exps, axis=1, keepdims=True)


def d_softmax(v_softmax):
	return v_softmax * (1 - v_softmax)
