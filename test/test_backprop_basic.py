import unittest

from ann.act import tanh, sigmoid
from ann.base import Network, FC
from ann.loss import cross_entropy_binary
from ann.opt import *


class ANNBinaryClassification(unittest.TestCase):
	# values from deeplearning.ai first course

	def test_forward(self):
		fc1 = FC(2, 4, tanh)
		fc2 = FC(4, 1, sigmoid)
		net = Network([fc1, fc2])

		# init parameters
		fc1.w = np.array([[-0.00416758, -0.00056267],
						  [-0.02136196, 0.01640271],
						  [-0.01793436, -0.00841747],
						  [0.00502881, -0.01245288]]).T
		fc1.b = np.array([[1.74481176], [-0.7612069], [0.3190391], [-0.24937038]]).T
		fc2.w = np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]).T
		fc2.b = np.array([[-1.3]]).T

		# propagate example
		x = np.array([[1.62434536, - 0.61175641, - 0.52817175],
					  [-1.07296862, 0.86540763, - 2.3015387]]).T
		net.forward(x)

		# check results
		self.assertAlmostEqual(np.mean(fc1.z), 0.262818640198)
		self.assertAlmostEqual(np.mean(fc1.a), 0.091999045227)
		self.assertAlmostEqual(np.mean(fc2.z), -1.30766601287)
		self.assertAlmostEqual(np.mean(fc2.a), 0.212877681719)

	def test_cost(self):
		fc = FC(3, 1, tanh)
		net = Network([fc])
		net.initialize()

		# init parameters
		y_pred = np.array([[0.5002307, 0.49985831, 0.50023963]]).T

		# compute cost example
		y = np.array([[True, False, False]]).T
		cost, _ = GradientDescent(cross_entropy_binary, 0).compute_loss(net, y_pred, y)

		# check results
		self.assertAlmostEqual(cost, 0.693058761)

	def test_backward(self):
		fc1 = FC(2, 4, tanh)
		fc2 = FC(4, 1, sigmoid)
		net = Network([fc1, fc2])

		# init parameters
		fc1.w = np.array([[-0.00416758, -0.00056267],
						  [-0.02136196, 0.01640271],
						  [-0.01793436, -0.00841747],
						  [0.00502881, -0.01245288]]).T
		fc1.b = np.array([[0], [0], [0], [0]]).T
		fc2.w = np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]).T
		fc2.b = np.array([[0]]).T

		# propagate example
		x = np.array([[1.62434536, - 0.61175641, - 0.52817175],
					  [-1.07296862, 0.86540763, - 2.3015387]]).T

		# check forward
		y_pred = net.forward(x)
		expected_y_pred = np.array([[0.5002307, 0.49985831, 0.50023963]]).T
		np.testing.assert_almost_equal(expected_y_pred, y_pred)

		# compute loss
		y = np.array([[True, False, True]]).T
		loss, dloss = GradientDescent(cross_entropy_binary, 0).compute_loss(net, y_pred, y)

		# propagate backwards
		net.backward(dloss)

		# check results
		expected_dw1 = np.array([[0.00301023, -0.00747267],
								 [0.00257968, -0.00641288],
								 [-0.00156892, 0.003893],
								 [-0.00652037, 0.01618243]]).T
		expected_db1 = np.array([[0.00176201], [0.00150995], [-0.00091736], [-0.00381422]]).T
		expected_dw2 = np.array([[0.00078841, 0.01765429, -0.00084166, -0.01022527]]).T
		expected_db2 = np.array([[-0.16655712]]).T
		np.testing.assert_almost_equal(fc1.dw, expected_dw1)
		np.testing.assert_almost_equal(fc1.db, expected_db1)
		np.testing.assert_almost_equal(fc2.dw, expected_dw2)
		np.testing.assert_almost_equal(fc2.db, expected_db2)
		# chain rule when sigmoid in last layer and cross entropy binary is just sigmoid activation minus labels
		np.testing.assert_almost_equal(fc2.dz, y_pred - y)


if __name__ == "__main__":
	unittest.main()
