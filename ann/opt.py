import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import shuffle

import ann.act
import ann.loss


class GradientDescent(object):
	def __init__(self, loss_func, lr, lr_decay=0, batch_size=0):
		self.loss_func = loss_func
		self.lr = lr
		self.lr_decay = lr_decay
		self.batch_size = batch_size
		self._its = 0

	def step(self, net, x, y):
		self._its += 1
		y_pred = net.forward(x, deterministic=False)
		loss, dy_pred = self.compute_loss(net, y_pred, y)
		net.backward(dy_pred)
		self.update_net(net)
		return loss

	def compute_loss(self, net, y_pred, y, no_d=False):
		loss = self.loss_func(y_pred, y)
		loss += self._get_weight_decay(net, y_pred)
		dy_pred = ann.loss.get_d_loss(self.loss_func)(y_pred, y) if not no_d else None
		return loss, dy_pred

	def update_net(self, net):
		lr = self.get_lr()
		for i, layer in enumerate(net.layers):
			layer.w -= lr * layer.dw
			layer.b -= lr * layer.db

	def optimize(self, net, x_train, y_train, epochs, track_loss=False, x_dev=None, y_dev=None, early_stop=False,
				 patience=100, tol=1e-8, iter_callback=None, verbose=0):
		self.reset()
		net.initialize()
		ls_train, ls_dev = [], []
		comp_loss_dev = x_dev is not None and y_dev is not None
		its_pat = 0
		if verbose:
			print("Starting optimization")
		for e in range(1, epochs + 1):
			for x_b, y_b in self.get_batches(x_train, y_train):
				loss_train = self.step(net, x_b, y_b)
				if comp_loss_dev:
					loss_dev, _ = self.compute_loss(net, net.forward(x_dev), y_dev, no_d=True)
				if verbose:
					max_its = int(np.rint(x_train.shape[0] / self.batch_size) if self.batch_size else 1 * epochs)
					print("Iteration {}/{}, loss train: {}".format(self._its, max_its, loss_train)
						  + (", loss dev: {}".format(loss_dev) if comp_loss_dev else ""))
				if np.math.isnan(loss_train) or np.math.isinf(loss_train) or comp_loss_dev and \
						(np.math.isinf(loss_dev) or np.math.isnan(loss_dev)):
					if verbose:
						print("Under-/overflow detected")
					break
				if early_stop:
					if ls_dev and loss_dev >= ls_dev[-1] - tol:
						its_pat += 1
					if its_pat == patience:
						if verbose:
							print("Stopping early")
						break
				if track_loss:
					ls_train.append(loss_train)
					if comp_loss_dev:
						ls_dev.append(loss_dev)
				if iter_callback:
					iter_callback(x_b, y_b, loss_train, loss_dev, self._its)
			else:
				continue
			if verbose:
				print("Optimization stopped")
			break
		else:
			if verbose:
				print("Optimization finished")
		return ls_train, ls_dev, self._its

	def get_batches(self, x, y):
		bs = self.batch_size or x.shape[0]
		nb = x.shape[0] // bs
		x, y = shuffle(x, y)
		if nb == 0:
			print("Warning: Batch size too big, using full dataset as batch")
			yield x, y
		elif nb == 1:
			yield x, y
		else:
			for i in range(nb):
				yield x[i * bs:(i + 1) * bs, :], y[i * bs:(i + 1) * bs, :]
			if x.shape[0] % bs != 0:
				yield x[nb * bs:, :], y[nb * bs:, :]

	def reset(self):
		self._its = 0

	def get_lr(self):
		return 1 / (1 + self.lr_decay * self._its) * self.lr

	@staticmethod
	def _get_weight_decay(net, y_pred):
		m = y_pred.shape[0]
		return 1 / (2 * m) * np.sum(
			[layer.weight_decay * np.linalg.norm(layer.w, "fro") if layer.weight_decay else 0 for layer in net.layers])


class GradientDescentMomentum(GradientDescent):
	def __init__(self, loss_func, lr, lr_decay, batch_size, m=0.9):
		super().__init__(loss_func, lr, lr_decay, batch_size)
		self.m = m

	def update_net(self, net):
		lr = self.get_lr()
		for i, layer in enumerate(net.layers):
			# update moments
			layer.mdw = self.m * layer.mdw + (1 - self.m) * layer.dw
			layer.mdb = self.m * layer.mdb + (1 - self.m) * layer.db
			# bias correction
			mdw = layer.mdw / (1 - np.power(self.m, self._its))
			mdb = layer.mdb / (1 - np.power(self.m, self._its))
			# update parameters
			layer.w -= lr * mdw
			layer.b -= lr * mdb


class RMSprop(GradientDescent):
	def __init__(self, loss_func, lr, lr_decay=0, batch_size=0, beta=0.999, eps=1e-8):
		super().__init__(loss_func, lr, lr_decay, batch_size)
		self.beta = beta
		self.eps = eps

	def update_net(self, net):
		lr = self.get_lr()
		for i, layer in enumerate(net.layers):
			# update moments
			layer.vdw = self.beta * layer.vdw + (1 - self.beta) * np.power(layer.dw, 2)
			layer.vdb = self.beta * layer.vdb + (1 - self.beta) * np.power(layer.db, 2)
			# bias correction
			vdw = layer.vdw / (1 - np.power(self.beta, self._its))
			vdb = layer.vdb / (1 - np.power(self.beta, self._its))
			# update parameters
			layer.w -= lr * np.divide(layer.dw, np.square(vdw) + self.eps)
			layer.b -= lr * np.divide(layer.db, np.square(vdb) + self.eps)


class Adam(GradientDescent):
	def __init__(self, loss_func, lr=0.001, lr_decay=0, batch_size=0, beta1=0.9, beta2=0.999, eps=1e-8):
		super().__init__(loss_func, lr, lr_decay, batch_size)
		self.beta1 = beta1
		self.beta2 = beta2
		self.eps = eps

	def update_net(self, net):
		lr = self.get_lr()
		beta1 = self.beta1
		beta2 = self.beta2
		for i, layer in enumerate(net.layers):
			# update moments
			layer.mdw = beta1 * layer.mdw + (1 - beta1) * layer.dw
			layer.mdb = beta1 * layer.mdb + (1 - beta1) * layer.db
			layer.vdw = beta2 * layer.vdw + (1 - beta2) * np.power(layer.dw, 2)
			layer.vdb = beta2 * layer.vdb + (1 - beta2) * np.power(layer.db, 2)
			# bias correction
			mdw = layer.mdw / (1 - np.power(beta1, self._its))
			mdv = layer.mdb / (1 - np.power(beta1, self._its))
			vdw = layer.vdw / (1 - np.power(beta2, self._its))
			vdb = layer.vdb / (1 - np.power(beta2, self._its))
			# update parameters
			layer.w -= lr * np.divide(mdw, np.square(vdw) + self.eps)
			layer.b -= lr * np.divide(mdv, np.square(vdb) + self.eps)
