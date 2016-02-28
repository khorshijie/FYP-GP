class InputReader:

	def __init__(self, file_name):
		self.file_name = file_name

	def get_data(self):
		X = []
		y = []
		with open(self.file_name) as f:
			content = f.readlines()
		for s in content:
			args = s.split();
			X.append(args[:-1])
			y.append(int(args[-1]))
		X = [[int(x) for x in lst] for lst in X]
		
		# for scalar inputs
		X = [item for sublist in X for item in sublist]

		return X, y


