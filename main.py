import read_input
import GPy
from matplotlib import pyplot as plt

# options for kernels: WN, C, LIN, SE, PER

if __name__ == "__main__":
	reader = read_input.InputReader("data/data.txt")
	X,y = reader.get_data("GPy")

	# Specify kernel
	kernel = GPy.kern.Whitenoise(1, 1)

	# Create GP model
	m = GPy.models.GPRegression(X,y,kernel)
	# m.optimize()
	# m.optimize_restarts(num_restarts = 10)
	# m.plot()
	
	plt.show()