import unittest

import numpy as np

from ann.loss import binary_xentropy


class LossFunctions(unittest.TestCase):

	def test_binary_cross_entropy(self):
		# four samples in a 3-class multi-label setting
		y = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1]])
		# sample activations
		a = np.array([[0.2, 0.31, 0.9],
					  [0.51, 0.01, 0.5],
					  [0.23, 0.75, 0.95],
					  [0.1, 0.55, 0.12]])

		loss = binary_xentropy(a, y)

		a_y = [0.2, 0.69, 0.1, 0.51, 0.01, 0.5, 0.23, 0.25, 0.05, 0.1, 0.55, 0.12]
		nll = -np.log(a_y)
		xentropy = np.sum(nll) / a.shape[0]

		self.assertAlmostEqual(loss, xentropy)
