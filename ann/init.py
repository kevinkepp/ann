import numpy as np

import ann.act


def normal(n_in, n_out, factor=0.01):
	return np.random.randn(n_in, n_out) * factor


def xavier(n_in, n_out):
	return np.random.randn(n_in, n_out) * np.sqrt(1. / n_out)


def relu(n_in, n_out):
	return np.random.randn(n_in, n_out) * np.sqrt(2. / n_out)


def bengio(n_in, n_out):
	return np.random.randn(n_in, n_out) * np.sqrt(2. / (n_out + n_in))


def get_default_init_func(activation):
	if activation == ann.act.relu:
		return relu
	else:
		return xavier
