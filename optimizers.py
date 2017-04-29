import numpy as np


class SGD(object):
    def __init__(self, learning_rate, decay=None):
        self.eta = learning_rate
        if decay:
            self.decay = None

    def update(self, param):
        """Update parameter values
        """
        param.value += - self.eta * param.grad


class RMSProp(object):
    def __init__(self, learning_rate, eps=10 ** -6, rho=0.5):
        self.eta = learning_rate
        self.eps = eps
        self.rho = rho

    def update(self, param):
        """Update parameter values
        :type param: gjnn.parameter
        """
        param.grad_hist = self.rho * param.grad_hist + (1 - self.rho) * (param.grad ** 2)
        param.value += -self.eta * param.grad / (np.sqrt(param.grad_hist) + self.eps)


class Adam(object):
    """arXiv:1412.6980"""

    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=10 ** -6):
        self.eta = learning_rate
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps

    def update(self, param):

        param.m = self.b1 * param.m + (1 - self.b1) * param.grad
        param.v = self.b2 * param.v + (1 - self.b2) * (param.grad ** 2)
        param.value += - self.eta * param.m / (np.sqrt(param.v) + self.eps)
