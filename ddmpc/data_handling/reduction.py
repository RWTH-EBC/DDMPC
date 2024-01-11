from abc import ABC, abstractmethod

import numpy as np
from sklearn import kernel_approximation
from sklearn.gaussian_process.kernels import Kernel, RBF


class InducingPoints(ABC):
    """
    The idea is to reduce the effective number of input data points x to the GP
    from n to m, with m<n, where the set of m points are called inducing points.
     Since this makes the effective covariance matrix K smaller,
     many inducing point approaches reduce the computational complexity from O(n3) to O(nm2).
     The smaller m is, the bigger the speed up.

     Source: https://bwengals.github.io/inducing-point-methods-to-speed-up-gps.html
    """

    def __init__(self):

        pass

    @abstractmethod
    def reduce(
            self,
            x:                      np.ndarray,
            y:                      np.ndarray,
            plot_distance_matrix:   bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:

        pass


