import read_input
import gaussian_process

# options for kernels: WN, C, LIN, SE, PER

if __name__ == "__main__":
	reader = read_input.InputReader("data.txt")
	X,y = reader.get_data()
	print X, y
	gp = gaussian_process.GaussianProcess(X, y, "LIN", [3, 0], 0.00001)
	print gp.predict(9)

