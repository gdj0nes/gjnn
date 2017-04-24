class layer:
    """
    Module is a super class. It could be a single layer, or a multilayer perceptron.
    """

    def __init__(self):
        return

    def forward(self, _input):
        """
        
        :param _input: 
        :return: 
        """
        pass

    def backward(self, _input, _gradOutput):
        """
        
        :param _input: 
        :param _gradOutput: 
        :return: 
        """
        pass

    def update(self, optimizer):
        pass
