import numpy as np
from GPy.kern import Kern

class Whitenoise(Kern):
    """
    Whitenoise Kernel
    """
    def __init__(self, dim, variance=1):
        self.input_dim = dim
        self.var = variance

    def _get_params(self):
    	return np.hstack(self.variance)