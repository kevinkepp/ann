import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import shuffle

import ann.act
import ann.loss


class SGD(object):
	def __init__(self, loss, lr=0.01, lr_decay=None, batch_size=None):
		self.loss = loss
		self.lr = lr
		self.lr_decay = lr_decay
		self.batch_size = batch_size
		self._its = 0

	def step(self, net, x, y):
		self._its += 1
		a = net.forward(x, deterministic=False)
		loss, da = self.compute_loss(net, a, y)
		net.backward(da)
		self.update(net)
		return loss

	def compute_loss(self, net, a, y, no_d=False):
		if y.ndim == 1:
			y = y.reshape(-1, 1)
		cost = self.loss(a, y)
		cost += self._get_weight_decay(net, a)
		da = ann.loss.get_d_loss(self.loss)(a, y) if not no_d else None
		return cost, da

	def update(self, net):
		lr = self.get_lr()
		for layer in net.layers:
			layer.vars["w"] -= lr * layer.vars["dw"]
			layer.vars["b"] -= lr * layer.vars["db"]

	def _init(self, net):
		self.reset()
		net.reset()
		net.initialize()

	def optimize(self, net, x_train, y_train, epochs, track_loss=False, x_dev=None, y_dev=None, early_stop_pat=None,
				 early_stop_tol=1e-8, iter_callback=None, break_on_overflow=True, verbose=0):
		self._init(net)
		ls_batch, ls_dev = [], []
		comp_loss_dev = x_dev is not None and y_dev is not None
		its_pat = 0
		if verbose:
			print("Starting optimization")
		for _ in range(epochs):
			for x_batch, y_batch in self.get_batches(x_train, y_train):
				loss_batch = self.step(net, x_batch, y_batch)
				if comp_loss_dev:
					loss_dev, _ = self.compute_loss(net, net.forward(x_dev), y_dev, no_d=True)
				if verbose:
					max_its = int(np.rint(x_train.shape[0] / self.batch_size) if self.batch_size else 1) * epochs
					print("Iteration {}/{}, loss batch: {}".format(self._its, max_its, loss_batch)
						  + (", loss dev: {}".format(loss_dev) if comp_loss_dev else ""))
				if np.math.isnan(loss_batch) or np.math.isinf(loss_batch) or comp_loss_dev and \
						(np.math.isinf(loss_dev) or np.math.isnan(loss_dev)):
					if verbose:
						print("Under-/overflow detected")
					if break_on_overflow:
						break
				if early_stop_pat:
					if ls_dev and loss_dev >= ls_dev[-1] - early_stop_tol:
						its_pat += 1
					if its_pat == early_stop_pat:
						if verbose:
							print("Stopping early")
						break
				if track_loss:
					ls_batch.append(loss_batch)
					if comp_loss_dev:
						ls_dev.append(loss_dev)
				if iter_callback:
					iter_callback(x_batch, y_batch, loss_batch, loss_dev, self._its)
			else:
				continue
			if verbose:
				print("Optimization stopped")
			break
		else:
			if verbose:
				print("Optimization finished")
		return ls_batch, ls_dev, self._its

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
				xb = x[i * bs:(i + 1) * bs, :]
				yb = y[i * bs:(i + 1) * bs, :] if y.ndim == 2 else y[i * bs:(i + 1) * bs, ]
				yield xb, yb
			if x.shape[0] % bs != 0:
				xb = x[nb * bs:, :]
				yb = y[nb * bs:, :] if y.ndim == 2 else y[nb * bs:, ]
				yield xb, yb

	def reset(self):
		self._its = 0

	def get_lr(self):
		return 1 / (1 + self.lr_decay * self._its) * self.lr if self.lr_decay else self.lr

	@staticmethod
	def _get_weight_decay(net, a):
		m = a.shape[0]
		return 1 / (2 * m) * np.sum(
			[layer.weight_decay * np.linalg.norm(layer.w, "fro") if layer.weight_decay else 0 for layer in net.layers])


class SGDM(SGD):
	def __init__(self, loss, lr=0.01, lr_decay=0, batch_size=0, m=0.9):
		super().__init__(loss, lr, lr_decay, batch_size)
		self.m = m

	def _init(self, net):
		super(SGDM, self)._init(net)
		for layer in net.layers:
			layer.vars["mdw"] = np.zeros(layer.vars["w"].shape)
			layer.vars["mdb"] = np.zeros(layer.vars["b"].shape)

	def update(self, net):
		lr = self.get_lr()
		for layer in net.layers:
			# update moments
			layer.vars["mdw"] = self.m * layer.vars["mdw"] + (1 - self.m) * layer.vars["dw"]
			layer.vars["mdb"] = self.m * layer.vars["mdb"] + (1 - self.m) * layer.vars["db"]
			# bias correction
			mdw = layer.vars["mdw"] / (1 - np.power(self.m, self._its))
			mdb = layer.vars["mdb"] / (1 - np.power(self.m, self._its))
			# update parameters
			layer.vars["w"] -= lr * mdw
			layer.vars["b"] -= lr * mdb


class RMSprop(SGD):
	def __init__(self, loss, lr=0.001, lr_decay=0, batch_size=0, rho=0.9, eps=1e-8):
		super().__init__(loss, lr, lr_decay, batch_size)
		self.rho = rho
		self.eps = eps

	def _init(self, net):
		super(RMSprop, self)._init(net)
		for layer in net.layers:
			layer.vars["vdw"] = np.zeros(layer.vars["w"].shape)
			layer.vars["vdb"] = np.zeros(layer.vars["b"].shape)

	def update(self, net):
		lr = self.get_lr()
		for layer in net.layers:
			# update moments
			layer.vars["vdw"] = self.rho * layer.vars["vdw"] + (1 - self.rho) * np.power(layer.vars["dw"], 2)
			layer.vars["vdb"] = self.rho * layer.vars["vdb"] + (1 - self.rho) * np.power(layer.vars["db"], 2)
			# bias correction
			vdw = layer.vars["vdw"] / (1 - np.power(self.rho, self._its))
			vdb = layer.vars["vdb"] / (1 - np.power(self.rho, self._its))
			# update parameters
			layer.vars["w"] -= lr * np.divide(layer.vars["dw"], np.square(vdw) + self.eps)
			layer.vars["b"] -= lr * np.divide(layer.vars["db"], np.square(vdb) + self.eps)


class Adam(SGD):
	def __init__(self, loss, lr=0.001, lr_decay=0, batch_size=0, beta1=0.9, beta2=0.999, eps=1e-8):
		super().__init__(loss, lr, lr_decay, batch_size)
		self.beta1 = beta1
		self.beta2 = beta2
		self.eps = eps

	def _init(self, net):
		super(Adam, self)._init(net)
		for layer in net.layers:
			layer.vars["mdw"] = np.zeros(layer.vars["w"].shape)
			layer.vars["mdb"] = np.zeros(layer.vars["b"].shape)
			layer.vars["vdw"] = np.zeros(layer.vars["w"].shape)
			layer.vars["vdb"] = np.zeros(layer.vars["b"].shape)

	def update(self, net):
		lr = self.get_lr()
		beta1 = self.beta1
		beta2 = self.beta2
		for layer in net.layers:
			# update moments
			layer.vars["mdw"] = beta1 * layer.vars["mdw"] + (1 - beta1) * layer.vars["dw"]
			layer.vars["mdb"] = beta1 * layer.vars["mdb"] + (1 - beta1) * layer.vars["db"]
			layer.vars["vdw"] = beta2 * layer.vars["vdw"] + (1 - beta2) * np.power(layer.vars["dw"], 2)
			layer.vars["vdb"] = beta2 * layer.vars["vdb"] + (1 - beta2) * np.power(layer.vars["db"], 2)
			# bias correction
			mdw = layer.vars["mdw"] / (1 - np.power(beta1, self._its))
			mdv = layer.vars["mdb"] / (1 - np.power(beta1, self._its))
			vdw = layer.vars["vdw"] / (1 - np.power(beta2, self._its))
			vdb = layer.vars["vdb"] / (1 - np.power(beta2, self._its))
			# update parameters
			layer.vars["w"] -= lr * np.divide(mdw, np.square(vdw) + self.eps)
			layer.vars["b"] -= lr * np.divide(mdv, np.square(vdb) + self.eps)
