import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from collections import Counter
import prettytable as pt
from PIL import Image
from scipy.stats import multivariate_normal



def import_csv(X_csv='dataset/gp_x.csv', T_csv='dataset/gp_t.csv'):
    X = pd.read_csv(X_csv).values.flatten()
    y = pd.read_csv(T_csv).values.flatten()
    '''data = dict()
    data['x'] = df['X'].values.reshape(100, 1)
    data['t'] = df['T'].values.reshape(100, 1)'''
    return X, y

def Read_JPG(filename):
    '''
    :param filename:  file path
    :return: matrix of RGB which type is numpy
    '''
    # be ware cv2 is read bgr
    img = Image.open(filename)

    return img

class gaussianProcess():
    def __init__(self, iter=100):
        self.parm =[
            #[16, 16, 16, 16],
            [0, 0, 0, 1],
            [1, 16, 0, 0],
            [1, 16, 0, 4],
            [1, 64, 32, 0],

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

    def predict(self, k, C_inv, t):
        return k.T.dot(C_inv).dot(t)

    def plot_gp(self, train_X, train_y, x, y, y1, y2, title, save=0):
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

    def loglikeihood(self, C_inv, C_dev, t):
        return -0.5 * np.trace(C_inv.dot(C_dev)) + 0.5 * np.linalg.multi_dot([t.T, C_inv, C_dev, C_inv, t])

    def ARD(self, train_X, train_y, params, split=50):
        parameters_ard = [[3, 6, 4, 5]]
        dev_func = [0, 0, 0, 0]
        learning_rate = 0.001
        while True:
            C_inv = np.linalg.inv(self.exponential_quadratic_kernel(train_X, train_X, params[-1]) + self.beta_inv * np.identity(split))

            # update parameter
            dev_func[0] = self.loglikeihood(C_inv, np.exp(-0.5 * params[-1][1] * np.subtract.outer(train_X, train_X) ** 2),
                                       train_y)

            dev_func[1] = self.loglikeihood(C_inv, params[-1][0] * -0.5 * np.subtract.outer(train_X, train_X) * np.exp(
                -0.5 * self.parm[-1][1] * np.subtract.outer(train_X, train_X) ** 2), train_y)

            dev_func[2] = self.loglikeihood(C_inv, np.full([split, split], 1), train_y)
            dev_func[3] = self.loglikeihood(C_inv, np.multiply.outer(train_X, train_X), train_y)
            params.append([p + learning_rate * dev for p, dev in zip(params[-1], dev_func)])

            if np.max(np.abs(dev_func)) < 6:
                return params


    def process(self, X, y, iter=500, split=50):
        train_X = X[:split]; test_X = X[split-1:]
        train_y = y[:split]; test_y = y[split-1:]
        x, y, y1, y2 = self.init_xy(iter)

        # iterate 4 parm
        for p in self.parm:
            C_inv = np.linalg.inv(self.exponential_quadratic_kernel(train_X, train_X, p) + self.beta_inv * np.identity(split))

            for i in range(iter):
                k = self.exponential_quadratic_kernel(train_X, x[i], p)
                c = self.exponential_quadratic_kernel(x[i], x[i], p) + self.beta_inv
                # mean
                y[i] = self.mean(k, C_inv, train_y)
                std = np.sqrt(self.cov(c, k, C_inv, train_y))
                y1[i] = y1[i] + std
                y2[i] = y2[i] - std
            # plot
            self.plot_gp(train_X, train_y, x, y, y1, y2, p)

            # cal rms on train/test data
            train_pred = np.empty(50)
            for i in range(50):
                k = self.exponential_quadratic_kernel(train_X, train_X[i], p)
                train_pred[i] = self.predict(k, C_inv, train_y)
            test_pred = np.empty(50)
            for i in range(50):
                k = self.exponential_quadratic_kernel(train_X, test_X[i], p)
                test_pred[i] = self.predict(k, C_inv, train_y)

            print('MSE result ', str(p), ':\n',
                  'training loss= ', self.RMSE(train_y, train_pred),
                  '\ntesting loss= ', self.RMSE(test_y, test_pred))

        # ARD
        parameters_ard = [[16, 16, 16, 16]]
        params_ard = self.ARD(train_X, train_y, parameters_ard)
        C_inv_ard = np.linalg.inv(self.exponential_quadratic_kernel(train_X, train_X, params_ard[-1]) + self.beta_inv * np.identity(split))
        for i in range(iter):
            k = self.exponential_quadratic_kernel(train_X, x[i], params_ard[-1])
            c = self.exponential_quadratic_kernel(x[i], x[i], params_ard[-1]) + self.beta_inv
            # mean
            y[i] = self.mean(k, C_inv_ard, train_y)
            std = np.sqrt(self.cov(c, k, C_inv_ard, train_y))
            y1[i] = y1[i] + std
            y2[i] = y2[i] - std
            # plot
        self.plot_gp(train_X, train_y, x, y, y1, y2, params_ard[-1])

def PCA(features, n=2, n_eigVect=None, typ=1):
    mean = np.mean(features.T, axis=1)
    center = features - mean
    cov = np.cov(features, rowvar=0)
    eigenvalue, eigenvector = np.linalg.eig(np.mat(cov))
    sortEigValue = np.argsort(eigenvalue)  # sort eigenvalue
    topNvalue = sortEigValue[-1:-(n + 1):-1]  # select top n value
    if typ ==1 : # training data
        n_eigVect = eigenvector[:, topNvalue]  # select largest n eigenvector
    # recon = (C*n_eigVect.T) + M  # reconstruct to original data
    Trans = center.dot(n_eigVect)  # transform to low dim data (same as the return of sklearn fit_transform())
    # transform matrix to array
    Trans = np.asarray(Trans).real
    return Trans, n_eigVect, sortEigValue
    # print(Trans.shape)
def normalize(X):
    return (X - X.mean())/X.std()

class supportVectorMachine():
    def __init__(self, type_='linear', C=1):
        self.type_ = type_
        self.class_label = [(0, 1), (0, 2), (1, 2)]
        self.C = C
        self.coef = None
        self.sv_index = None

    def fit(self, X, y, vs='ovo', c=1):
        if self.type_ == 'linear':
            clf = SVC(kernel=self.type_, C=c, decision_function_shape=vs)
        else:
            clf = SVC(kernel=self.type_, C=c, degree=2, decision_function_shape=vs)

        clf.fit(X, y)
        self.coef = np.abs(clf.dual_coef_)
        self.sv_index = clf.support_

    def linear_phi(self, x):
            return x

    def poly_phi(self, x):
        if len(x.shape) == 1:
            return np.vstack((x[0] ** 2, np.sqrt(2) * x[0] * x[1], x[1] ** 2)).T
        else:
            return np.vstack((x[:, 0] ** 2, np.sqrt(2) * x[:, 0] * x[:, 1], x[:, 1] ** 2)).T

    def kernel_function(self, xn, xm):
        if self.type_ == 'linear':
            return np.dot(self.linear_phi(xn), self.linear_phi(xm).T)
        else:
            return np.dot(self.poly_phi(xn), self.poly_phi(xm).T)

    def prepare_parameter_for_classifiers(self, X):
        size=100
        # target
        target_dict = {}
        target_dict[(0, 1)] = np.concatenate((np.ones(size), np.full([size], -1), np.zeros(size)))
        target_dict[(0, 2)] = np.concatenate((np.ones(size), np.zeros(size), np.full([size], -1)))
        target_dict[(1, 2)] = np.concatenate((np.zeros(size), np.ones(size), np.full([size], -1)))
        # multiplier
        multiplier = np.zeros([len(X), 2])
        multiplier[self.sv_index] = self.coef.T

        multiplier_dict = {}
        multiplier_dict[(0, 1)] = np.concatenate((multiplier[:size * 2, 0], np.zeros(size)))
        multiplier_dict[(0, 2)] = np.concatenate((multiplier[:size, 1], np.zeros(size), multiplier[size * 2:, 0]))
        multiplier_dict[(1, 2)] = np.concatenate((np.zeros(size), multiplier[size:, 1]))
        print(multiplier_dict)
        return target_dict, multiplier_dict

    def get_w_b(self, a, t, x):
        # PRML 7.29, 7.37
        at = a * t
        if self.type_ == 'linear':
            w = at.dot(self.linear_phi(x))
        else:
            w = at.dot(self.poly_phi(x))

        M_indexes = np.where(((a > 0) & (a < self.C)))[0]
        S_indexes = np.nonzero(a)[0]
        Nm = len(M_indexes)

        if Nm == 0:
            b = -1
        else:
            # PRML 7.18 7.37
            # b = 1 / Ns *Sum(tn - Sum(a*t*kernel))
            b = np.mean(t[M_indexes] - at[S_indexes].dot(self.kernel_function(x[M_indexes], x[S_indexes]).T))

        return w, b

    def train(self, X):
        target_dict, multiplier_dict = self.prepare_parameter_for_classifiers(X)
        weight_dict = {}
        bias_dict = {}

        for c1, c2 in self.class_label:
            weight, bias = self.get_w_b(multiplier_dict[(c1, c2)], target_dict[(c1, c2)], X)
            weight_dict[(c1, c2)] = weight
            bias_dict[(c1, c2)] = bias
        return weight_dict, bias_dict

    def predict(self, X, weight_dict, bias_dict):
        pred = []
        for index in range(len(X)):
            votes = []
            for c1, c2 in self.class_label:
                weight = weight_dict[(c1, c2)]
                bias = bias_dict[(c1, c2)]
                if self.type_ == 'linear':
                    y = weight.dot(self.linear_phi(X[index]).T) + bias
                else:
                    y = weight.dot(self.poly_phi(X[index]).T) + bias
                if y > 0:
                    votes += [c1]
                else:
                    votes += [c2]
            pred += [Counter(votes).most_common()[0][0]]
        return pred

    def plot(self, X, t, xx, yy, prediction):
        class0_indexes = np.where(t == 0)
        class1_indexes = np.where(t == 1)
        class2_indexes = np.where(t == 2)
        plt.scatter(X[self.sv_index, 0], X[self.sv_index, 1], facecolors='none', edgecolors='k', linewidths=2,
                    label="support vector")
        plt.scatter(X[class0_indexes][:, 0], X[class0_indexes][:, 1], c='r', marker='x', label="class 0")
        plt.scatter(X[class1_indexes][:, 0], X[class1_indexes][:, 1], c='g', marker='*', label="class 1")
        plt.scatter(X[class2_indexes][:, 0], X[class2_indexes][:, 1], c='b', marker='^', label="class 2")
        plt.legend()

        plt.contourf(xx, yy, prediction, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.show()

    def make_meshgrid(self, x, y, h=0.02):
        space = 0.3
        x_min, x_max = x.min() - space, x.max() + space
        y_min, y_max = y.min() - space, y.max() + space
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy

    def process(self, X_csv='dataset/x_train.csv', T_csv='dataset/t_train.csv'):

        X = pd.read_csv(X_csv, header=None).values
        y = pd.read_csv(T_csv, header=None).iloc[:, 0].values
        # using PCA reduce dim
        X, comp, _ = PCA(X)
        X = normalize(X)
        # linear
        linear_svm = supportVectorMachine()
        linear_svm.fit(X, y)
        weight_dict, bias_dict = linear_svm.train(X)
        xx, yy = linear_svm.make_meshgrid(X[:, 0], X[:, 1])
        prediction = linear_svm.predict(np.column_stack((xx.flatten(), yy.flatten())),
                                 weight_dict,
                                 bias_dict)
        linear_svm.plot(X, y, xx, yy, np.array(prediction).reshape(xx.shape))


        # poly
        poly_svm = supportVectorMachine(type_='poly')
        poly_svm.fit(X, y)
        weight_dict, bias_dict = poly_svm.train(X)
        xx, yy = poly_svm.make_meshgrid(X[:, 0], X[:, 1])
        prediction = poly_svm.predict(np.column_stack((xx.flatten(), yy.flatten())),
                                        weight_dict,
                                        bias_dict)
        poly_svm.plot(X, y, xx, yy, np.array(prediction).reshape(xx.shape))


class kmeans():
    def __init__(self, data, k=2, max_iter=300):
        self.k = k
        self.max_iter = max_iter
        self.eye = np.eye(k)
        # we need optimal mu
        # mu size (k, rgb)
        self.mu = data[np.random.choice(len(data), self.k, replace=False)]
        # if k = argmin(norm(x-mu)**2)
        # else 0
        # rnk size (x*y, k)
        self.rnk = np.ones([len(data), self.k])

    def minimize_j(self, data):
        for i in range(self.max_iter):
            distance = np.sum((data[:, None] - self.mu)**2, axis=2)
            # pixel -> pixels*k
            rnk = self.eye[np.argmin(distance, axis=1)]

            if np.array_equal(rnk, self.rnk):
                break
            else:
                self.rnk = rnk
            self.mu = np.sum(rnk[:, :, None] * data[:, None], axis=0) / np.sum(rnk, axis=0)[:, None]

    def print_rgb_table(self, _type):
        tb = pt.PrettyTable()
        tb.add_column(_type, [k for k in range(self.k)])
        tb.add_column('R', [r for r in (self.mu[:, 0] * 255).astype(int)])
        tb.add_column('G', [g for g in (self.mu[:, 1] * 255).astype(int)])
        tb.add_column('B', [b for b in (self.mu[:, 2] * 255).astype(int)])
        print("======= K = %d (%s) =======" % (self.k, _type))
        print(tb)
        print()

    def generate_image(self, _type):
        if _type == 'K_means':
            new_data = (self.mu[np.where(self.rnk == 1)[1]] * 255).astype(int)
        else:
            new_data = (self.mu[np.argmax(self.gaussians, axis=0)] * 255).astype(int)

        disp = Image.fromarray(new_data.reshape(height, width, depth).astype('uint8'))
        # disp.show(title=_type)
        disp.save(_type + str(k) + '.jpg')

class gaussianMixtureModel():
    def __init__(self, k, k_mean_rnk, k_mean_mu, data, lr=1e-7, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.pi = np.sum(k_mean_rnk, axis=0) / len(k_mean_rnk)
        # k *rgb*rgb
        self.cov = np.array([np.cov(data[np.where(k_mean_rnk[:, k] == 1)[0]].T) for k in range (self.k)])
        self.lr = lr
        self.gaussian = np.array([ multivariate_normal.pdf(data, mean=k_mean_mu[k], cov=self.cov[k])*self.pi[k] for k in range(self.k) ])
        self.loss = []
    def stepE(self):
        '''
        Expectation
        :return:
        '''
        self.gamma = (self.gaussian / np.sum(self.gaussian, axis=0)).T
        
    def stepM(self, data):
        '''
        Maximization
        :param data: X
        :return:
        '''
        nk = np.sum(self.gamma, axis=0)
        self.mu = np.sum(self.gamma[:, :, None]*data[:, None], axis=0) / nk[:, None]
        for k in range(self.k):
            self.cov[k] = (self.gamma[:, k, None] * (data - self.mu[k])).T.dot(data - self.mu[k]) / nk[
                k] + self.lr * np.eye(depth)
            self.pi = nk / len(data)

    def evaulate(self, data):
        for k in range(self.k):
            self.gaussian[k] = multivariate_normal.pdf(data, mean=self.mu[k], cov=self.cov[k])*self.pi[k]
        self.loss.append(self.log_likeihood())

    def log_likeihood(self):
        return np.sum(np.log(np.sum(self.gaussian, axis=0)))

    def em_algorithm(self, data):

        for i in range(self.max_iter):
            self.stepE()
            self.stepM(data)
            self.evaulate(data)

    def plot_likelihood_log(self):
        plt.title('Log likelihood of GMM (k=%d)' % self.k)
        plt.plot([i for i in range(100)], self.loss)
        plt.savefig('log_likelihood_' + str(self.k) + '.png')
        plt.show()



K_list = [3, 5, 7, 10]
img = Image.open('dataset/imghw3.jpg')
img.load()
data = np.asarray(img, dtype='float')/255
height, width, depth = data.shape
data = np.reshape(data, (-1, depth)) # pixels*RGB  (width*height) = pixels
for k in K_list:
    k_mean = kmeans(data, k=k)
    k_mean.minimize_j(data)
    #k_mean.print_rgb_table('K_means')
    #k_mean.generate_image('K_means')
    gmm = gaussianMixtureModel(k, k_mean.rnk, k_mean.mu, data)
    gmm.em_algorithm(data)
    gmm.plot_likelihood_log()

