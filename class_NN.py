import numpy
import scipy.special # for sigmoid: expit()

class neuralNetwork: 
    
    #initialize the NN
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        """
        the number of input, hidden and output layer nodes can be defined by parameters. 
        in this function, they are defined so that they can be used later.
        also, the weights (from input to hidden, from hidden to output) are initialized.
        
        :param inputnodes: specify the number of input nodes 
        :param hiddennodes: specify the number of hidden nodes 
        :param outputnodes: specify the number of output nodes (equal to the number of possible output labels)
        :param learningrate: specify the learning rate 
        
        """
        # input, hidden and output nodes
        self.input = inputnodes
        self.hidden = hiddennodes
        self.output = outputnodes
        
        ### WEIGHT INITIALIZATION ### 
        # set the normal distribution to 0, and the standard deviation to 
        # the number of hidden nodes to the power of -0.5 
        # a random number following this distribution is taken for each weight.
        
        # weights input-hidden
        self.wih = numpy.random.normal(0.0, pow(self.hidden, -0.5), 
                               (self.hidden, self.input))
        
        # weights hidden-input
        self.who = numpy.random.normal(0.0, pow(self.output, -0.5), 
                               (self.output, self.hidden))
        
        # learning rate
        self.lr = learningrate
        
        # activation function (sigmoid)
        # this non-linear function maps the input to a value between 0 and 1. 
        # when the resulting value is below 0.5, the neuron will not fire a signal.
        # when the resulting value is higher than 0.5, the neuron will fire a signal.
        self.activation_function = lambda x: scipy.special.expit(x)

        
    # train the NN
    def train(self, inputs_list, targets_list):
        """
        Calculate outputs, compare them to error and propagate the information back
        
        Throughout this function, matrix multiplication is used to make fast calculations
        for e.g. the input/output signals into each layer (where the network input matrix 
        is multiplied with the matrix containing the weights), and the updating of weights 
        (where error matrices are multiplied). See the report on this project for the formulas. 
        
        :param inputs_list: contains inputs 
        :param targets_list: contains gold data
        
        """
        ### convert inputs to a matrix 
        # matrix multiplication is used in this nn to make 
        # fast calculations for input/output signals, 
        # and updating weights. 
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        ### HIDDEN LAYER 
        # calculate signal into hidden layer
        # by taking the dot-product of the network inputs and weights
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the outputs from hidden layer 
        hidden_outputs = self.activation_function(hidden_inputs)
        
        ### OUTPUT LAYER 
        # calculate signals into output layer
        # by taking the dot-product of the hidden outputs and weights
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate outputs of final layer by applying sigmoid function to layer inputs
        final_outputs = self.activation_function(final_inputs)
        
        
        ### calculate errors 
        # this is simply the difference between the predicted value and target value
        output_errors =  targets - final_outputs
        # hidden layer errors: output_errors, 
        # here, the errors should be split corresponding to the weights that go into the neuron,
        # where high weights receive a large portion of the error. see report for a more elaborate explanation
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        ### UPDATING WEIGHTS 
        # hidden - output 
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), 
                                        numpy.transpose(hidden_outputs))
        # input - hidden 
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), 
                                        numpy.transpose(inputs))
    
    # query the NN
    def query(self, inputs_list):
        """
        Takes the input and returns the predicted labels (output)
        The output takes the form of an array containing probabilities for each label
        The highest probability corresponds to the predicted label

        :param inputs_list: inputs_list
        :return: final_outputs 
        
        """
        # convert inputs to 2d array (matrix)
        inputs = numpy.array(inputs_list, ndmin=2).T 
        
        ### HIDDEN LAYER
        # calculate signal into the hidden layer 
        # by taking the dot-product of the network inputs and weights
        hidden_inputs = numpy.dot(self.wih, inputs)
        # apply sigmoid function
        hidden_outputs = self.activation_function(hidden_inputs)
        
        ### OUTPUT LAYER
        # calculate dot product weights and input
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # apply sigmoid function
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs 