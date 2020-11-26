import numpy as np

'''class GaussianFeature(object):
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

    def _gauss_basis(self, x, mean, width, axis=None):
        
        arg = (x-mean) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))
    
    def fit(self, X):
        self.centers_ = np.linspace(X.min(), X.max(), self.N) # mu
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0]) # s
        
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
'''
        
class GaussianFeature(object):
    """
    Gaussian feature
    gaussian function = exp(-0.5 * (x - m) / v)
    """

    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def _gauss(self, x, mean):
        return np.exp(-0.5 * np.sum(np.square(x - mean), axis=-1) / self.var)

    def transform(self, x):
        basis = [np.ones(len(x))]
        for m in self.mean:
            basis.append(self._gauss(x, m))
        return np.asarray(basis).transpose()