import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer

from ann.act import relu, softmax_with_cross_entropy
from ann.loss import cross_entropy_with_softmax
from ann.opt import SGDM
from ann.sklearn import NetworkClassifier, FC

# set seeds
random.seed(42)
np.random.seed(42)

# prepare data
mnist = fetch_mldata('MNIST original')
x = StandardScaler().fit_transform(mnist.data)
y = LabelBinarizer().fit_transform(mnist.target.astype(int))
x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=1000, stratify=y)

# define model and optimization
layers = [
	FC(n_in=x.shape[1], n_out=1024, act=relu),
	FC(n_in=1024, n_out=1024, act=relu),
	FC(n_in=1024, n_out=10, act=softmax_with_cross_entropy)
]
opt = SGDM(loss_func=cross_entropy_with_softmax, lr=0.01, batch_size=64, m=0.9)
net = NetworkClassifier(layers, opt)
epochs = 5

# prepare plots
fig, axs = plt.subplots(2, 1, sharex=True)
ax1, ax2 = axs
fig.show()
ls_batch, ls_dev, accs_batch, accs_dev = [], [], [], []
plot_every = 10


# define optimization callback to draw plots
def callback(x_batch, y_batch, loss_batch, loss_dev, its):
	ls_batch.append(loss_batch)
	ls_dev.append(loss_dev)
	# compute accuracies
	acc_batch = net.score(x_batch, y_batch)
	acc_dev = net.score(x_dev, y_dev)
	accs_batch.append(acc_batch)
	accs_dev.append(acc_dev)
	# refresh plot
	if its % plot_every == 0:
		steps = range(its)
		ax1.clear()
		ax1.plot(steps, ls_batch, label="Batch {:.3f}".format(loss_batch))
		ax1.plot(steps, ls_dev, label="Dev {:.3f}".format(loss_dev))
		ax1.set_ylabel("Loss")
		ax1.legend()
		ax2.clear()
		ax2.plot(steps, accs_batch, label="Batch {:.3f}".format(accs_batch[-1]))
		ax2.plot(steps, accs_dev, label="Dev {:.3f}".format(accs_dev[-1]))
		ax2.set_xlabel("Iterations")
		ax2.set_ylim([0, 1])
		ax2.set_ylabel("Accuracy")
		ax2.legend()
		fig.canvas.draw()


# train network and plot results
net.fit(x_train, y_train, epochs=epochs, x_dev=x_dev, y_dev=y_dev, iter_callback=callback, verbose=1)

# keep figure open
plt.show()
