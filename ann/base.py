import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

import ann.act
import ann.init
import ann.loss


class Module(object):
	def initialize(self):
		pass

	def forward(self, v_in, deterministic=True):
		pass

	def backward(self, d_in):
		pass

	def update(self, rate):
		pass


class ANN(Module):
	def __init__(self, loss, d_loss):
		self.layers = []

	def add_node(self, n):
		self.layers += [n]
		return self

	def initialize(self):
		for n in self.layers:
			n.initialize()

	def forward(self, v_in, deterministic=True):
		v_out = None
		for n in self.layers:
			v_out = n.forward(v_in, deterministic)
			v_in = v_out
		return v_out



	def backward(self, d_in):
		d_out = None
		for n in reversed(self.layers):
			d_out = n.backward(d_in)
			d_in = d_out
		return d_out

	def update(self, learning_rate):
		for n in self.layers:
			n.update(learning_rate)

	def train(self, x, y, loss, learning_rate, epochs, stop_at_loss_change_rate=None, verbose=0):
		"""x.shape (features, nb_samples), y.shape (1, nb_samples)"""
		if x.shape[0] != self.layers[0].n_in:
			raise ValueError("Input dim {} wrong, expected {}".format(x.shape[0], self.layers[0].n_in))
		self.initialize()
		l_old = None
		for e in range(epochs):

			l_old = l
		if verbose:
			print("Done")

	def classify(self, x):
		"""x.shape (features, nb_samples)"""
		# assume one output unit and sigmoid activation
		return np.round(self.forward(x, deterministic=True))


class FullyConnected(Module):
	def __init__(self, n_in, n_out, init=ann.init.xavier, activation=ann.act.relu, weight_decay_factor=0,
				 dropout_rate=0):
		self.n_in = n_in
		self.n_out = n_out
		self.init = init
		self.activation = activation
		self.weight_decay_factor = weight_decay_factor
		self.dropout_rate = dropout_rate
		self.w = None
		self.b = None
		self.v_in = None
		self.dw = None
		self.db = None
		self.drop = None

	def initialize(self):
		self.w = self.init(self.n_out, self.n_in)
		self.b = np.zeros((self.n_out, 1))

	def forward(self, v_in, deterministic=True):
		self.v_in = v_in
		v_out = np.dot(self.w, v_in) + self.b
		if not deterministic and self.dropout_rate:
			self.drop = np.random.rand(v_out.shape[0], v_out.shape[1]) > self.dropout_rate
			v_out = np.multiply(v_out, self.drop)
			v_out /= self.dropout_rate
		return v_out

	def backward(self, d_in):
		m = self.v_in.shape[1]
		l2 = self.weight_decay_factor / m * self.w if self.weight_decay_factor else 0
		d_in = np.multiply(d_in, self.df(self.v_out))
		self.dw = np.dot(d_in, self.v_in.T) / m + l2
		self.db = np.sum(d_in, axis=1, keepdims=True) / m
		d_out = np.dot(self.w.T, d_in)
		return d_out

	def update(self, rate):
		self.w -= self.dw * rate
		self.b -= self.db * rate


class Nonlinearity(Layer):
	def __init__(self, func):
		if func == "relu":
			self.f = relu
			self.df = d_relu
		elif func == "tanh":
			self.f = np.tanh
			self.df = d_tanh
		elif func == "sigmoid":
			self.f = sigmoid
			self.df = d_sigmoid
		elif func == "softmax":
			self.f = softmax
			self.df = d_softmax
		else:
			raise NotImplementedError("activation function '{}' not supported".format(func))
		self.v_out = None

	def init(self):
		pass

	def forward(self, v_in, deterministic=True):
		self.v_out = self.f(v_in)
		return self.v_out

	def backward(self, d_in):

	def update(self, rate):
		pass


class ANNEstimator(ANN, BaseEstimator):
	def __init__(self, n_in, n_out, nonlinearity_out, hidden=None, loss=None, learning_rate=None, epochs=None,
				 stop_at_loss_change_rate=None, initializations=None, weight_decay_factors=None, dropout_rates=None,
				 verbose=0):
		super(ANNEstimator, self).__init__()
		self.n_in = n_in
		self.n_out = n_out
		self.nonlinearity_out = nonlinearity_out
		self.hidden = hidden
		self.loss = loss
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.stop_at_loss_change_rate = stop_at_loss_change_rate
		self.initializations = None
		self.weight_decay_factors = weight_decay_factors
		self.dropout_rates = dropout_rates
		self.verbose = verbose

	def _build(self):
		hidden = self.hidden or []
		n_prev = self.n_in
		for i, (n, nl) in enumerate(hidden + [(self.n_out, self.nonlinearity_out)]):
			init = self.initializations[i] if isinstance(self.initializations, (list, tuple)) else self.initializations
			wd_facotr = self.weight_decay_factors[i] if isinstance(self.weight_decay_factors,
																   (list, tuple)) else self.weight_decay_factors
			drop_rate = self.dropout_rates[i] if isinstance(self.dropout_rates, (list, tuple)) else self.dropout_rates
			self.add_node(FullyConnected(n_prev, n, init, wd_facotr, drop_rate))
			self.add_node(Nonlinearity(nl))
			n_prev = n

	def fit(self, x, y):
		self._build()
		x = x.T
		y = y.reshape(1, -1)
		super(ANNEstimator, self).train(x, y, self.loss, self.learning_rate, self.epochs, self.stop_at_loss_change_rate,
										self.verbose)

	def predict(self, x):
		x = x.T
		v = super(ANNEstimator, self).forward(x, deterministic=True)
		return v.T

	def get_params(self, deep=False):
		return {
			"n_in": self.n_in,
			"n_out": self.n_out,
			"nonlinearity_out": self.nonlinearity_out,
			"hidden": self.hidden,
			"loss": self.loss,
			"learning_rate": self.learning_rate,
			"epochs": self.epochs,
			"stop_at_loss_change_rate": self.stop_at_loss_change_rate,
			"initializations": self.initializations,
			"weight_decay_factors": self.weight_decay_factors,
			"dropout_rates": self.dropout_rates,
			"verbose": self.verbose
		}


class ANNClassifier(ANNEstimator, ClassifierMixin):
	def predict(self, x):
		# assume 1 output neuron and sigmoid activation
		x = x.T
		c = super(ANNClassifier, self).classify(x)
		return c.T

	def predict_proba(self, x):
		p = super(ANNClassifier, self).predict(x)
		ps = np.empty((x.shape[0], 2))
		ps[:, 0] = p[:, 0]
		ps[:, 1] = 1 - p[:, 0]
		return ps
