class layer:
    """
    Module is a super class. It could be a single layer, or a multilayer perceptron.
    """

    def __init__(self):
        self.train = None

    def forward(self, input_):
        """
        
        :type input_: object
        :return: 
        """
        pass

    def backward(self, input_, output_gradient):
        """
        
        :param input_: 
        :param output_gradient: 
        :return: 
        """
        pass

    def update(self, optimizer):
        pass

    def set_train(self):
        self.train = True

    def set_eval(self):
        self.train = False
