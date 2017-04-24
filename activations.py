import numpy as np
from gjnn.base import layer


class softmax(layer):
    '''Preform softmax on input
    '''

    def __init__(self):
        layer.__init__(self)

    def forward(self, scores):
        '''
        '''
        z = np.exp(scores).sum(axis=1, keepdims=True)
        probs = np.exp(scores) / z  # - np.log(np.sum(np.exp(input_), axis=1))

        return probs

    def backward(self, probs, labels):
        # print (labels)
        num_obs = len(probs)
        input_grad = probs.copy()
        input_grad[range(num_obs), labels.astype(int)] -= 1
        input_grad /= num_obs

        return input_grad

    def evaluate(self, probs, labels):
        '''Consider moving this to a cross entropy loss layer if possible'''
        num_obs = len(probs)
        class_logprob = -np.log(probs[np.arange(num_obs), labels.astype(int)])
        loss = np.sum(class_logprob) / num_obs

        return loss

class tanh(layer):
    def __init__(self):
        layer.__init__(self)

    def forward(self, input_):
        output = np.tanh(input_)
        return output

    def backward(self, layer_input, output_grad):
        input_grad = output_grad * (1 - np.square(np.tanh(layer_input)))
        return input_grad


class reLU(layer):
    def __init__(self):
        layer.__init__(self)

    def forward(self, input_):
        output = input_.copy()
        output[output < 0] = 0
        return output

    def backward(self, layer_input, output_grad):
        input_grad = output_grad.copy()
        input_grad[layer_input < 0] = 0
        return input_grad


class leakyReLU(layer):
    def __init__(self, alpha):
        layer.__init__(self)
        self.alpha = alpha

    def forward(self, input_):
        output = input_.copy()
        output[output < 0] = self.alpha * output[output < 0]
        return output

    def backward(self, layer_input, output_grad):
        input_grad = output_grad.copy()
        input_grad[layer_input < 0] = self.alpha * input_grad[layer_input < 0]
        return input_grad