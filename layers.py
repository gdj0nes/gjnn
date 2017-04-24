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
            self.weights.grad += (self.weights.value * self.l2) / self.size
            # self.bias.grad += self.bias.value * self.l2
        input_grad = np.dot(output_gradient, self.weights.value.T)

        return input_grad

    def update(self, optimizer):
        for param in self.params:
            param.update(optimizer)


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