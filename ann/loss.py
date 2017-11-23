import numpy as np

def cross_entropy_binary(v_out, v_true):
	v_out = np.maximum(1e-10, np.minimum(1 - 1e-10, v_out))
	m = v_true.shape[1]
	cost = -np.dot(np.log(v_out), v_true.T) / m - np.dot(np.log(1 - v_out), 1 - v_true.T) / m
	return np.squeeze(cost).item()


def d_cross_entropy_binary(v_out, v_true):
	v_out = np.maximum(1e-10, np.minimum(1 - 1e-10, v_out))
	return -(np.divide(v_true, v_out) - np.divide(1 - v_true, 1 - v_out))


def cross_entropy(v_out, v_true):
	v_out = np.maximum(1e-10, np.minimum(1 - 1e-10, v_out))
	m = v_true.shape[1]
	cost = -np.dot(np.log(v_out), v_true.T) / m
	return np.squeeze(cost).item()


def d_cross_entropy(v_out, v_true):
	v_out[np.abs(v_out) < 1e-10] = 1e-10
	return -np.divide(v_true, v_out)
