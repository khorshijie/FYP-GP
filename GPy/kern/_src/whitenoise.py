# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from .kern import Kern
from ...util.linalg import tdot
from ...core.parameterization import Param
from ...core.parameterization.transformations import Logexp
from ...util.caching import Cache_this
from ...util.config import *
from .psi_comp import PSICOMP_Linear

class Whitenoise(Kern):
    """
    Linear kernel

    .. math::

       k(x,y) = \sum_{i=1}^{\\text{input_dim}} \sigma^2_i x_iy_i

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variances: the vector of variances :math:`\sigma^2_i`
    :type variances: array or list of the appropriate size (or float if there
                     is only one variance parameter)
    :param ARD: Auto Relevance Determination. If False, the kernel has only one
                variance parameter \sigma^2, otherwise there is one variance
                parameter per dimension.
    :type ARD: Boolean
    :rtype: kernel object

    """

    def __init__(self, dim, variance=1,active_dims=None, name="whitenoise"):
        super(Whitenoise, self).__init__(dim, active_dims, name)
        self.input_dim = dim
        self.var = variance

    def _get_params(self):
        return np.hstack(self.variance)

    def _set_params(self, x):
        self.var = x[0]

    def _get_param_names(self):
        return ['variance']

    @Cache_this(limit=2)
    def K(self, X, X2=None):
        n = len(X)
        ans = np.zeros(shape=(n,n))
        if X2 == None:
            for i in range(n):
                for j in range(n):
                    if X[i] == X[j]:
                        ans[i,j] = self.var
                    else: 
                        ans[i,j] = 0
        else: 
            for i in range(n):
                for j in range(n):
                    if X[i] == X2[j]:
                        ans[i,j] = self.var
                    else:
                        ans[i,j] = 0
        return ans


    def Kdiag(self, X):
        mat = K(X)
        ans = 0
        n = len(X)
        for i in range(n):
            ans += mat[i,i]
        return ans

    #---------------------------------------#
    #             PSI statistics            #
    #---------------------------------------#

    def psi0(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[0]

    def psi1(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[1]

    def psi2(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[2]

    def psi2n(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior, return_psi2_n=True)[2]
