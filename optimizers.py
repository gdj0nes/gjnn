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

    def __init__(self, learning_rate, delta=10**-6, rho=0.5):
        self.eta = learning_rate
        self.delta = delta
        self.rho = rho

    def update(self, param):
        """Update parameter values
        :type param: gjnn.parameter
        """
        param.grad_hist = self.rho * param.grad_hist + (1 - self.rho) * (param.grad ** 2)
        param.value += -self.eta * param.grad / (np.sqrt(param.grad_hist) + self.delta)


class Adam(object):
    """TO BE IMPLEMENTED"""

    def __init__(self, learning_rate):
        self.eta = learning_rate

    def update(self, value, gradient):
        if gradient:
            value += - self.eta * gradient