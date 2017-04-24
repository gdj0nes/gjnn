"""
weights.py

Contains functions related to the creation of model weights/parameters

"""
import numpy as np


def xavier_init(shape):
    """Xavier initialization

    Citation: 
    """
    sq = np.sqrt(6.0 / np.prod(shape))
    return np.random.uniform(-sq, sq, shape)


class parameter(object):
    """Object for trainable parameter values"""

    def __init__(self, shape, initializer, name=None):
        """
        Parameters:
        ----------
        shape (tuple): The shape of the weight matrix
        initializer (function): Produces the model weights
        """

        self.value = initializer(shape)
        self.grad = None
        self.grad_hist = 0.0
        self.name = name

    def update(self, optimizer):
        """Update the value of the parameter using the model
        optimizer
        """
        optimizer.update(self)
