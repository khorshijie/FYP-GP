import math

def get_kernel(name, params):
	return Kernel(name, params)

class Kernel:
	def __init__(self, name, params):
		self.params = params
		self.name = name

	def cov(self, a, b):
		if self.name == "WN":
			return self.whitenoise(a,b)
		elif self.name == "C":
			return self.constant(a,b)
		elif self.name == "LIN":
			return self.linear(a,b)
		elif self.name == "SE":
			return self.square_exponential(a,b)
		elif self.name == "PER":
			return self.periodic(a,b)
		else:
			return 0

	# current versions work for scalar inputs only
	def whitenoise(self, a, b):
		if a == b:
			return self.params[0] ** 2
		else:
			return 0

	def constant(self, a,b):
		return self.params[0] ** 2

	def linear(self, a, b):
		return self.params[0] ** 2 * (a - self.params[1]) * (b - self.params[1])

	def square_exponential(self, a, b):
		return self.params[0] ** 2 * math.exp(- (a - b) ** 2 / 2 / self.params[1] ** 2)

	def periodic(self, a,b):
		return 0