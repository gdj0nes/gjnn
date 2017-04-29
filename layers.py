import numpy as np

from gjnn.weights import parameter
from gjnn.base import layer

class FullyConnected(layer):
    """Full Connected Layer"""

    def __init__(self, input_size, num_nodes, initializer, l2=None):
        layer.__init__(self)
        self.input_size = input_size
        self.size = num_nodes

        self.weights = parameter((self.input_size, self.size), initializer)
        self.bias = parameter((1, num_nodes), initializer)
        self.params = [self.weights, self.bias]
        self.l2 = l2

    def forward(self, input_):
        """
        out = X * W + b
        """

        output = np.dot(input_, self.weights.value) + self.bias.value
        return output

    def backward(self, input_, output_gradient):
        """

        """
        # Compute gradients
        self.weights.grad = np.dot(input_.T, output_gradient)
        self.bias.grad = output_gradient.sum(axis=0)
        if self.l2:
            self.weights.grad += (self.weights.value * self.l2)

        input_grad = np.dot(output_gradient, self.weights.value.T)

        return input_grad

    def update(self, optimizer):
        for param in self.params:
            param.update(optimizer)


class BatchNormalization(layer):

    def __init__(self, momentum=0.95, eps=0.001):
        layer.__init__(self)
        self.mom = momentum
        self.eps = eps
        self.mu = 0.0
        self.sigma = 0.0
        self.batch_sigma = 0.0
        self.batch_mu = 0.0

    def forward(self, input_):

        if self.train:

            self.batch_mu = input_.mean(axis=0)
            self.batch_sigma = np.sqrt(np.square(input_ - batch_mu).mean(axis=0) + self.eps)

            output = (input_ - batch_mu) / batch_sigma
            self.mu = self.mom * self.mu + (1 - self.mom) * batch_mu
            self.sigma = self.mom * self.mu + (1 - self.mom) * batch_sigma

        else:

            output = (input_ - self.mu) / self.sigma

        return output

    def backward(self, input_, output_gradient):
        pass

class DropOut(layer):
    """https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf"""

    def __init__(self, portion):
        layer.__init__(self)
        self.p = portion
        self.mask = None

    def forward(self, input_):

        if self.train:
            self.mask = np.random.binomial(1, self.p, input_.shape)
        else:
            self.mask = self.p

        return (input_ * self.mask) / (1 - self.p)

    def backward(self, input_, output_gradient):

        return (self.mask * output_gradient) / (1 - self.p)


class conv2D(layer):
    def __init__(self, kernel_shape, num_filters, initializer, padding=False):
        layer.__init__(self)
        self.padding = padding
        self.weights = parameter((kernel_shape[0], kernel_shape[0], num_filters), initializer)
        self.bias = parameter((1, num_filters), initializer)
        self.params = [self.weights, self.bias]

    def forward(self, input_):
        """Figure out how to single op forward pass
        """
        pass

    def backward(self, input_, output_gradient):
        """Figure out how to single op forward pass (is this a thing)
        """
        pass

    def update(self, optimizer):
        for param in self.params:
            param.update(optimizer)
