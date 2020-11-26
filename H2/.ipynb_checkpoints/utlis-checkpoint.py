import numpy as np
import math

def sigmoid(a):
    # σ(a) = 1 / (1 + exp(−a))
    return 1 / (1 + np.exp(-a))

class RegressionModel:
    def __init__(self, M=3, s=0.6): 
        self.M = M
        self.s = s
    
    def basis_function(self, x):
        # M: number of parameter
        # μj: govern the locations of the basis functions in input space
        # s: governs their spatial scale.
        M, s = self.M, self.s
        new_x = np.empty([len(x), 0])

        for j in range(0, M):
            mu = 2*j/M
            new_x = np.concatenate((new_x, sigmoid((x-mu)/s)), axis=1)
        return new_x
    
    def posterior_distrib(self, x, t, alpha=math.pow(10, -6), beta=1):
        M, s = self.M, self.s
        
        phi = self.basis_function(x)
        # Calculation variance first
        # S0 = alpha*I
        # Sn = S0^-1 + beta*phi.T.dot(phi)
        S0 = alpha*np.eye(M)
        Sn_inverse = S0 + beta*phi.T.dot(phi)
        Sn = np.linalg.inv(Sn_inverse)
        #mN = beta*Sn*phi.T*t
        mN = beta*Sn.dot(phi.T).dot(t)
        return mN, Sn
        
    def predict(self, x, mN, Sn):
        w = np.random.multivariate_normal(mN.reshape(self.M), Sn, size=5)
        
        return x.dot(w.T)
    
    def predict_distrib(self, x, mN, SN, beta=1):
        phi = self.basis_function(x)
        mean = phi.dot(mN)
        var = 1/beta + np.sum(phi.dot(SN.dot(phi.T)))
        SD = np.sqrt(var)
        return mean.reshape(len(x)), SD