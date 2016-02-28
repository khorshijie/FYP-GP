import read_input
import gaussian_process
import GPy
from matplotlib import pyplot as plt

# options for kernels: WN, C, LIN, SE, PER

if __name__ == "__main__":
	reader = read_input.InputReader("data.txt")
	X,y = reader.get_data("GPy")
	print X, y

	# Specify linear kernel
	kernel = GPy.kern.Linear(1)

	# Create GP model
	m = GPy.models.GPRegression(X,y,kernel)
	m.optimize()
	m.optimize_restarts(num_restarts = 10)
	m.plot()
	
	plt.show()