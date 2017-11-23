import math
import random

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelBinarizer

from ann.act import relu, softmax
from ann.loss import cross_entropy
from ann.opt import Adam
from ann.sklearn import NetworkClassifier, FC

# set seeds
random.seed(42)
np.random.seed(42)

# prepare data
mnist = fetch_mldata('MNIST original')
x = mnist.data
x = StandardScaler().fit_transform(x)
y = mnist.target.astype(int)
y = LabelBinarizer().fit_transform(y)
print("MNIST dataset: X.shape {}, Y.shape {}".format(x.shape, y.shape))
dev_size = 200
x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=dev_size, stratify=y)
x_train, x_dev, y_train, y_dev = x_train.T, x_dev.T, y_train.T, y_dev.T

# define model and optimization
net = NetworkClassifier() \
	.add(FC(x.shape[1], 1024, relu)) \
	.add(FC(1024, 1024, relu)) \
	.add(FC(1024, 10, softmax))
opt = Adam(loss=cross_entropy, lr=5 * 1e-9, batch_size=512)

# train
epochs = 1
early_stop = True
pat = 20
ls_train, ls_dev, accs_batch, accs_dev = [], [], [], []
fig, axs = plt.subplots(2, 1, sharex=True)
ax1, ax2 = axs
fig.show()


def callback(x_b, y_b, l_train, l_dev, its):
	acc_batch = net.score(x_b.T, y_b.T)
	acc_dev = net.score(x_dev.T, y_dev.T)
	ls_train.append(l_train)
	ls_dev.append(l_dev)
	accs_batch.append(acc_batch)
	accs_dev.append(acc_dev)
	ax1.clear()
	ax1.plot(range(its), ls_train, label="Train {:.3f}".format(l_train))
	ax1.plot(range(its), ls_dev, label="Dev {:.3f}".format(l_dev))
	ax1.set_ylabel("Loss")
	ax1.legend()
	ax2.clear()
	ax2.plot(range(its), accs_batch, label="Batch {:.3f}".format(acc_batch))
	ax2.plot(range(its), accs_dev, label="Dev {:.3f}".format(acc_dev))
	ax2.set_xlabel("Iterations")
	ax2.set_ylim([0, 1])
	ax2.set_ylabel("Accuracy")
	ax2.legend()
	fig.canvas.draw()
	plt.draw()


opt.optimize(net, x_train, y_train, epochs, x_dev=x_dev, y_dev=y_dev, iter_callback=callback, verbose=1)
plt.show()
