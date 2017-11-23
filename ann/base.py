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


class FC(object):
	"""Fully connected layer"""

	def __init__(self, n_in, n_out, act, init=None, weight_decay=0, dropout=0):
		self.n_in = n_in
		self.n_out = n_out
		self.act = act
		self.d_act = ann.act.get_d_act(act)
		self.init = init if init else ann.init.get_default_init_func(act)
		self.weight_decay = weight_decay
		self.dropout = dropout
		self.a_prev = None
		self.w = None
		self.b = None
		self.z = None
		self.a = None
		self.drop = None
		self.da = None
		self.dz = None
		self.dw = None
		self.db = None
		self.da_prev = None
		self.mdw = None
		self.mdb = None
		self.vdw = None
		self.vdb = None

	def initialize(self):
		self.w = self.init(self.n_in, self.n_out)
		self.b = np.zeros((1, self.n_out))
		self.mdw = np.zeros(self.w.shape)
		self.mdb = np.zeros(self.b.shape)
		self.vdw = np.zeros(self.w.shape)
		self.vdb = np.zeros(self.b.shape)

	def forward(self, a_prev, deterministic=True):
		self.a_prev = a_prev
		self.z = np.dot(a_prev, self.w) + self.b
		self.a = self.act(self.z)
		if not deterministic and self.dropout:
			self.drop = np.random.rand(self.a.shape[0], self.a.shape[1]) > self.dropout
			self.a = np.multiply(self.a, self.drop)
			self.a /= (1 - self.dropout)
		return self.a

	def backward(self, da):
		self.da = da
		if self.drop is not None:
			da = np.multiply(da, self.drop)
			da /= (1 - self.dropout)
		m = self.a_prev.shape[0]
		l2 = self.weight_decay / m * self.w
		if self.act == ann.act.softmax:
			# from https://stackoverflow.com/a/33580680/1662053
			# TODO gradient check and understand this
			tmp = np.multiply(self.a, da)
			s = np.sum(tmp, axis=1, keepdims=True)
			self.dz = tmp - self.a * s
		elif self.act == ann.act.softmax_with_cross_entropy:
			self.dz = da  # dz directly calculated in d_cross_entropy_with_softmax
		else:
			self.dz = np.multiply(da, self.d_act(self.a))
		self.dw = np.dot(self.a_prev.T, self.dz) / m + l2
		self.db = np.sum(self.dz, axis=0, keepdims=True) / m
		self.da_prev = np.dot(self.dz, self.w.T)
		return self.da_prev

	def __repr__(self):
		return "FC(n_in={}, n_out={}, act={}, init={}, weight_decay={}, dropout={})".format(
			self.n_in, self.n_out, self.act.__name__, self.init.__name__, self.weight_decay, self.dropout)
