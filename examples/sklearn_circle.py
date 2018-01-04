import random
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# prepare data
from ann.act import sigmoid, relu
from ann.base import FC
from ann.loss import cross_entropy_binary
from ann.opt import GradientDescent
from ann.sklearn import NetworkClassifier

# set seeds
random.seed(42)
np.random.seed(42)

# prepare data
x, y = make_circles()
x = StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# define model and optimization
layers = [
	FC(n_in=x_train.shape[1], n_out=32, act=relu),
	FC(n_in=32, n_out=1, act=sigmoid)
]
opt = GradientDescent(loss_func=cross_entropy_binary, lr=0.1)
net = NetworkClassifier(layers, opt)

# train network
t_start = time.time()
losses, _, _ = net.fit(x_train, y_train, epochs=5000, track_loss=True, verbose=1)
t_dur = time.time() - t_start
print("Training took {:.3f} seconds".format(t_dur))
print("Final loss: {:.5f}".format(losses[-1]))

# evaluate network
x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
Z = net.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
x_test_pred = net.predict(x_test)
test_acc = accuracy_score(y_test, x_test_pred)

# plot results
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
plt.figure(figsize=(14, 7))
ax = plt.subplot(1, 2, 1)
ax.set_title("Training")
ax.plot(range(len(losses)), losses)
ax.set_xlabel("iterations")
ax.set_ylabel("loss")
ax = plt.subplot(1, 2, 2)
ax.set_title("Evaluation")
ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train.reshape(-1, ), cmap=cm_bright, edgecolors='k')
ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test.reshape(-1, ), cmap=cm_bright, edgecolors='k', alpha=0.6)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
ax.text(xx.max() - .3, yy.min() + .3, ('Test acc: %.2f' % test_acc).lstrip('0'), size=12, horizontalalignment='right')
plt.tight_layout()
plt.show()
