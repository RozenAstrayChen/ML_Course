import numpy as np

class GaussianFeature(object):
    """
    Gaussian feature
    gaussian function = exp(-0.5 * (x - m) / v)
    """

    def __init__(self, N, width_factor=1.0):
        """
        construct gaussian features
        Parameters
        ----------
        N: 
        """
        self.N = N
        self.width_factor = width_factor

    def _gauss_basis(self, x, y, width, axis=None):
        arg = (x-y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))
    
    def fit(self, X):
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        
    def transform(self, x):
        """
        transform input array with gaussian features
        Parameters
        ----------
        x : (sample_size, ndim) or (sample_size,)
            input array
        Returns
        -------
        output : (sample_size, n_features)
            gaussian features
        """
        return self._gauss_basis(x[:,: ,np.newaxis], self.centers_,
                                 self.width_, axis=1)
