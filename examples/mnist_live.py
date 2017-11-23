import math
import random

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelBinarizer

from ann.act import relu, softmax_with_cross_entropy, softmax
from ann.loss import cross_entropy, cross_entropy_with_softmax
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
dev_size = 1000
x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=dev_size, stratify=y)

# define model and optimization
layers = [
	FC(x.shape[1], 1024, relu),
	FC(1024, 1024, relu),
	FC(1024, 10, softmax_with_cross_entropy)
]
opt = Adam(loss_func=cross_entropy_with_softmax, lr=5 * 1e-9, batch_size=512)
net = NetworkClassifier(layers, opt)
epochs = 1
early_stop = True
pat = 20

# prepare plots
fig, axs = plt.subplots(2, 1, sharex=True)
ax1, ax2 = axs
fig.show()
ls_train, ls_dev, accs_batch, accs_dev = [], [], [], []


# define callback to draw plots
def callback(x_b, y_b, loss_train, loss_dev, its):
	# TODO improve performance
	ls_train.append(loss_train)
	ls_dev.append(loss_dev)
	acc_batch = net.score(x_b, y_b)
	acc_dev = net.score(x_dev, y_dev)
	accs_batch.append(acc_batch)
	accs_dev.append(acc_dev)
	ax1.clear()
	ax1.plot(range(its), ls_train, label="Train {:.3f}".format(loss_train))
	ax1.plot(range(its), ls_dev, label="Dev {:.3f}".format(loss_dev))
	ax1.set_ylabel("Loss")
	ax1.legend()
	ax2.clear()
	ax2.plot(range(its), accs_batch, label="Batch {:.3f}".format(accs_batch[-1]))
	ax2.plot(range(its), accs_dev, label="Dev {:.3f}".format(accs_dev[-1]))
	ax2.set_xlabel("Iterations")
	ax2.set_ylim([0, 1])
	ax2.set_ylabel("Accuracy")
	ax2.legend()
	fig.canvas.draw()


# train network and plot results
net.fit(x_train, y_train, epochs=epochs, x_dev=x_dev, y_dev=y_dev, iter_callback=callback, verbose=1)

# keep figure open
plt.show()
