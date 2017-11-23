from sklearn.base import BaseEstimator, ClassifierMixin

from ann.base import *


class NetworkEstimator(Network, BaseEstimator):
	def __init__(self, layers=None, opt=None):
		super(NetworkEstimator, self).__init__(layers)
		self.opt = opt

	def fit(self, x, y, **fit_args):
		res = self.opt.optimize(self, x, y, **fit_args)
		return res

	def predict(self, x):
		return super(NetworkEstimator, self).forward(x, deterministic=True)


class NetworkClassifier(NetworkEstimator, ClassifierMixin):
	def predict(self, x):
		p = self.predict_proba(x)
		if p.shape[1] == 2:
			return np.argmax(p, axis=1)
		else:
			c = np.zeros(p.shape)
			c[range(p.shape[0]), np.argmax(p, axis=1)] = 1
			return c

	def predict_proba(self, x):
		a = self.forward(x, deterministic=True)
		if a.shape[1] == 1:
			a = np.hstack((1 - a, a))
		return a
