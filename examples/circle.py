import time

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# prepare data
from ann.act import sigmoid, relu
from ann.base import FullyConnected
from ann.loss import cross_entropy_binary
from ann.opt import GradientDescent
from ann.sklearn import NetworkClassifier

np.random.seed(42)

x, y = make_circles()
print("Dataset: X.shape {}, Y.shape {}".format(x.shape, y.shape))
x = StandardScaler().fit_transform(x)
y = y.astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# train my ANN
layers = [FullyConnected(x_train.shape[1], 128, relu), FullyConnected(128, 1, sigmoid)]
net = NetworkClassifier(layers, optimizer_cls=GradientDescent, epochs=1000, verbose=1,
						optimizer__loss=cross_entropy_binary, optimizer__lr=0.5, optimizer__batch_size=0)

t_start = time.time()
losses = net.fit(x_train, y_train)
t_dur = time.time() - t_start
print("Training took {} seconds".format(t_dur))
print("Final loss: {}".format(losses[-1]))

# prepare mesh
x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

plt.figure(figsize=(16, 8))
ax = plt.subplot(1, 2, 1)
Z = net.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
score = net.score(x_test, y_test)
ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')
ax.set_title("Network")

ax = plt.subplot(1, 2, 2)
ax.plot(range(len(losses)), losses)
ax.set_xlabel("iterations")
ax.set_ylabel("loss")

plt.tight_layout()
plt.show()
