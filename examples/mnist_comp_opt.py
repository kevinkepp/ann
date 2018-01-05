import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer

from ann.act import relu, softmax_with_xentropy
from ann.loss import xentropy_with_softmax
from ann.opt import SGD, RMSprop, Adam, SGDM
from ann.sklearn import NetworkClassifier, FC

# set seeds
random.seed(42)
np.random.seed(42)

# prepare data
mnist = fetch_mldata('MNIST original')
x = mnist.data
y = LabelBinarizer().fit_transform(mnist.target.astype(int))
x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=1000, stratify=y)
# normalize input
scaler = StandardScaler(copy=False)
x_train = scaler.fit_transform(x_train)
x_dev = scaler.transform(x_dev)

# define configurations
configs = []

net1 = NetworkClassifier(layers=[
	FC(n_in=x.shape[1], n_out=256, act=relu),
	FC(n_in=256, n_out=10, act=softmax_with_xentropy)
])
opt1 = SGD(loss=xentropy_with_softmax, lr=0.001, batch_size=64)
configs.append(("SGD", net1, opt1))

net2 = net1.clone()
opt2 = SGDM(loss=xentropy_with_softmax, lr=0.001, batch_size=64)
configs.append(("SGDM", net2, opt2))

net3 = net1.clone()
opt3 = RMSprop(loss=xentropy_with_softmax, lr=1e-8, batch_size=64)
configs.append(("RMSprop", net3, opt3))

net4 = net1.clone()
opt4 = Adam(loss=xentropy_with_softmax, lr=1e-8, batch_size=64)
configs.append(("Adam", net4, opt4))

# --- you can add other configurations here ---

# define training procedure
epochs = 10
early_stop_patience = 200

# train networks
results = []
for _, net, opt in configs:
	res = opt.optimize(net, x_train, y_train, epochs, x_dev=x_dev, y_dev=y_dev, track_loss=True,
					   early_stop_pat=early_stop_patience, verbose=1)
	results.append(res)


def plot(ax, ls_batch, ls_dev, its, title):
	ax.plot(range(len(ls_batch)), ls_batch, label="Batch")
	ax.plot(range(len(ls_dev)), ls_dev, label="Dev")
	ax.text(0.3, 0.93, "Batch: {:.3f}".format(ls_batch[-1]), transform=ax.transAxes)
	ax.text(0.3, 0.86, "Dev: {:.3f}".format(ls_dev[-1]), transform=ax.transAxes)
	ax.text(0.3, 0.79, "Its: {}".format(its), transform=ax.transAxes)
	ax.set_xlabel("Iterations")
	ax.set_ylabel("Loss")
	ax.set_title(title)
	ax.legend(loc="upper right")


# plot results
rows = np.sqrt(len(configs)).astype(np.int)
cols = np.ceil(len(configs) / rows).astype(np.int)
plt.figure(figsize=(4 * cols, 4 * rows))
last_ax = None
for i, ((title, net, opt), (ls_batch, ls_dev, its)) in enumerate(zip(configs, results)):
	ax = plt.subplot(rows, cols, i + 1, sharex=last_ax, sharey=last_ax)
	if len(ls_batch) > 0:
		plot(ax, ls_batch, ls_dev, its, title)
	else:
		print("Warning: Config {} did not return any results".format(title))
	last_ax = ax
plt.tight_layout()
plt.show()
