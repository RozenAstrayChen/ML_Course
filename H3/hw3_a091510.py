import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def import_csv(X_csv='dataset/gp_x.csv', T_csv='dataset/gp_t.csv'):
    X = pd.read_csv(X_csv).values.flatten()
    y = pd.read_csv(T_csv).values.flatten()
    '''data = dict()
    data['x'] = df['X'].values.reshape(100, 1)
    data['t'] = df['T'].values.reshape(100, 1)'''
    return X, y

class gaussianProcess():
    def __init__(self, iter=100):
        self.parm =[
            [0, 0, 0, 1],
            [1, 16, 0, 0],
            [1, 16, 0, 4],
            [1, 64, 32, 0]
        ]

        self.beta_inv = 1
    def init_xy(self, iter):
        x = np.linspace(0, 2, iter)
        y = np.empty(iter)
        y1 = np.empty(iter)
        y2 = np.empty(iter)
        return x, y, y1, y2

    def exponential_quadratic_kernel(selfx, xn, xm, param):
        f = param[0]*np.exp(-0.5*param[1]*np.subtract.outer(xn, xm)**2)
        s = param[2]
        t = param[3]*np.multiply.outer(xn, xm) # I don't no dot will happen error
        return  f+s+t

    def mean(self, k, C_inv, t):
        #return np.linalg.multi_dot([k, C_inv, t])
        return k.T.dot(C_inv).dot(t)

    def cov(self, C, k, C_inv, t):
        return C - self.mean(k, C_inv, t)

    def plot_gp(self, train_X, train_y, x, y, y1, y2, title, save=1):
        plt.plot(x, y, 'r-')
        plt.fill_between(x, y1, y2, facecolor='pink', edgecolor='none')
        plt.scatter(train_X, train_y, facecolors='none', edgecolors='b')  # plot the data point
        plt.title(str(title))
        plt.xlim(0, 2)
        plt.ylim(-10, 15)
        plt.xlabel('x')
        plt.ylabel('t', rotation=0)
        if save:
            plt.savefig('.gp_param' + str(title) + '.png')
        plt.show()

    def RMSE(self, x, y):
        return np.sqrt(np.sum((x-y)**2)/len(x))

    def process(self, X, y, iter=100, split=50):
        train_X = X[:split]; test_X = X[split:]; train_y = y[:split]; test_y = y[split:]
        x, y, y1, y2 = self.init_xy(iter)

        # iterate 4 parm
        for p in range(4):
            C_inv = np.linalg.inv(self.exponential_quadratic_kernel(train_X, train_X, self.parm[p]) + self.beta_inv * np.identity(split))

            for i in range(iter):
                k = self.exponential_quadratic_kernel(train_X, x[i], self.parm[p])
                c = self.exponential_quadratic_kernel(x[i], x[i], self.parm[p]) + self.beta_inv
                # mean
                y[i] = self.mean(k, C_inv, train_y)
                std = np.sqrt(self.cov(c, k, C_inv, train_y))
                y1[i] = y1[i] + std
                y2[i] = y2[i] - std
            # plot
            self.plot_gp(train_X, train_y, x, y, y1, y2, self.parm[p])

X, y = import_csv()
gp = gaussianProcess()
gp.process(X, y)