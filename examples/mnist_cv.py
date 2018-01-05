import random

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from ann.act import relu, softmax_with_xentropy
from ann.loss import xentropy_with_softmax
from ann.opt import SGD
from ann.sklearn import NetworkClassifier, FC

# set seeds
random.seed(42)
np.random.seed(42)

# prepare dataset
mnist = fetch_mldata('MNIST original')
x = mnist.data
y = LabelBinarizer().fit_transform(mnist.target.astype(int))

# define model and optimization
opt = SGD(loss=xentropy_with_softmax, batch_size=64)
net = NetworkClassifier(opt=opt)
pipe = Pipeline([("norm", StandardScaler()), ("net", net)])

# execute cross-validation
param_dist = {
	"net__layers": [[FC(x.shape[1], 8, relu), FC(8, 10, softmax_with_xentropy)],
					[FC(x.shape[1], 64, relu), FC(64, 10, softmax_with_xentropy)],
					[FC(x.shape[1], 1024, relu), FC(1024, 10, softmax_with_xentropy)]],
	"net__opt__lr": np.logspace(-4, 0, num=5, base=10),

}
opt_fit_params = {"net__epochs": 5}
grid = RandomizedSearchCV(pipe, param_dist, n_jobs=-1, fit_params=opt_fit_params, verbose=1)
grid.fit(x, y)

# inspect results
for param, train, test in sorted(list(
		zip(grid.cv_results_["params"], grid.cv_results_["mean_train_score"], grid.cv_results_["mean_test_score"])),
		key=lambda s: s[2], reverse=True):
	print("n_hid {} with lr {}:\ttrain {:.4f}, test {:.4f}".format(param["net__layers"][0].n_out, param["net__opt__lr"],
																   train, test))
