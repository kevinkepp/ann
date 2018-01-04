import numpy as np

import ann.act
import ann.init
import ann.loss


class Network(object):
	def __init__(self, layers=None):
		self.layers = layers

	def add(self, n):
		self.layers = (self.layers or []) + [n]
		return self

	def initialize(self):
		for l in self.layers:
			l.initialize()

	def forward(self, x, deterministic=True):
		a_prev = x
		for l in self.layers:
			a = l.forward(a_prev, deterministic)
			a_prev = a
		return a_prev

	def backward(self, dy_pred):
		da_next = dy_pred
		for l in reversed(self.layers):
			d_a = l.backward(da_next)
			da_next = d_a
		return da_next

	def reset(self):
		for l in self.layers:
			l.reset()

	def clone(self):
		return Network([l.clone() for l in self.layers])


class FC(object):
	"""Fully connected layer"""

	def __init__(self, n_in, n_out, act, init=None, weight_decay=0, dropout=0):
		self.n_in = n_in
		self.n_out = n_out
		self.act = act
		self.d_act = ann.act.get_d_act(act)
		self.init = init
		self.weight_decay = weight_decay
		self.dropout = dropout
		self.vars = {}

	def initialize(self):
		init = self.init if self.init else ann.init.get_default_init_func(self.act)
		self.vars["w"] = init(self.n_in, self.n_out)
		self.vars["b"] = np.zeros((1, self.n_out))

	def forward(self, a_prev, deterministic=True):
		self.vars["a_prev"] = a_prev
		self.vars["z"] = np.dot(a_prev, self.vars["w"]) + self.vars["b"]
		self.vars["a"] = self.act(self.vars["z"])
		if not deterministic and self.dropout:
			self.vars["drop_mask"] = np.random.rand(self.vars["a"].shape[0], self.vars["a"].shape[1]) > self.dropout
			self.vars["a"] = np.multiply(self.vars["a"], self.vars["drop_mask"]) / (1 - self.dropout)
		return self.vars["a"]

	def backward(self, da):
		self.vars["da"] = da
		self.vars["da_drop"] = np.multiply(da, self.vars["drop_mask"]) / (1 - self.dropout) \
			if "drop_mask" in self.vars else da
		m = self.vars["a_prev"].shape[0]
		l2 = self.weight_decay / m * self.vars["w"]
		if self.act == ann.act.softmax:
			# from https://stackoverflow.com/a/33580680/1662053
			# TODO gradient check and understand this
			tmp = np.multiply(self.vars["a"], self.vars["da_drop"])
			s = np.sum(tmp, axis=1, keepdims=True)
			self.vars["dz"] = tmp - self.vars["a"] * s
		elif self.act == ann.act.softmax_with_cross_entropy:
			self.vars["dz"] = self.vars["da_drop"]  # dz directly calculated in loss.d_cross_entropy_with_softmax
		else:
			self.vars["dz"] = np.multiply(self.vars["da_drop"], self.d_act(self.vars["a"]))
		self.vars["dw"] = np.dot(self.vars["a_prev"].T, self.vars["dz"]) / m + l2
		self.vars["db"] = np.sum(self.vars["dz"], axis=0, keepdims=True) / m
		self.vars["da_prev"] = np.dot(self.vars["dz"], self.vars["w"].T)
		return self.vars["da_prev"]

	def reset(self):
		self.vars.clear()

	def clone(self):
		c = FC(self.n_in, self.n_out, self.act, self.init, self.weight_decay, self.dropout)
		c.vars = self.vars.copy()
		return c

	def __repr__(self):
		return "FC(n_in={}, n_out={}, act={}, init={}, weight_decay={}, dropout={})".format(
			self.n_in, self.n_out, self.act.__name__, self.init.__name__, self.weight_decay, self.dropout)
