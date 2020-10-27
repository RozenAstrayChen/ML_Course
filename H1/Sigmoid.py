import numpy as np


class SigmoidalFeature(object):
    """
    Sigmoidal features

    1 / (1 + exp((m - x) @ c)
    """

    def __init__(self, N, width_factor=1.0, coef=1):
        """
        construct gaussian features
        Parameters
        ----------
        N: 
        """
        self.N = N
        self.width_factor = width_factor

    def _sigmoid_basis(self, x, mean, width, axis=None):
        arg = np.sum((x-mean) / width, axis)
        
        return 1/(1+np.exp(-arg))
    
    def fit(self, X):
        self.centers_ = np.linspace(X.min(), X.max(), self.N) # mu
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0]) # s


    def transform(self, x):
        """
        transform input array with sigmoidal features

        Parameters
        ----------
        x : (sample_size, ndim) or (sample_size,) ndarray
            input array

        Returns
        -------
        output : (sample_size, n_features) ndarray
            sigmoidal features
        """
        return self._sigmoid_basis(x[:,: ,np.newaxis], self.centers_,
                                 self.width_, axis=1)
    