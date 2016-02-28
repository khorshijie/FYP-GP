import kernel
import numpy
from numpy.linalg import inv

class GaussianProcess:

	def __init__(self, X, y, kern, params, noise):
		self.n = len(y)
		self.X = X
		self.y = y
		self.noise = noise
		self.kern = kernel.get_kernel(kern, params)

	def predict(self, x):
		temp_klist = []
		for i in range(self.n):
			temp_klist.append(self.kern.cov(x, self.X[i]))
		k = numpy.matrix(numpy.asarray(temp_klist))
		y = numpy.matrix(numpy.asarray(self.y))
		K = numpy.empty([self.n, self.n])
		for i in range(self.n):
			for j in range(self.n):
				K[i,j] = self.kern.cov(self.X[i], self.X[j])
		expected = k * inv(numpy.multiply(self.noise, numpy.identity(self.n)) + K) * numpy.transpose(y)
		var = self.kern.cov(x, x) - k * inv(numpy.multiply(self.noise, numpy.identity(self.n)) + K) * numpy.transpose(k)
		return expected, var