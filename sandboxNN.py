from codecs import decode
import struct
from numpy import exp, array, random, dot, exp2, tanh, zeros, heaviside, transpose

def bin_to_float(b):
    """ Convert binary string to a float. """
    bf = int_to_bytes(int(b, 2), 8)  # 8 bytes needed for IEEE 754 binary64.
    return struct.unpack('>d', bf)[0]

def int_to_bytes(n, length):  # Helper function
    """ Int/long to byte string.

        Python 3.2+ has a built-in int.to_bytes() method that could be used
        instead, but the following works in earlier versions including 2.x.
    """
    return decode('%%0%dx' % (length << 1) % n, 'hex')[-length:]

def float_to_bin(value):  # For testing.
    """ Convert float to 64-bit binary string. """
    [d] = struct.unpack(">Q", struct.pack(">d", value))
    return '{:064b}'.format(d)





# Genes are in hex string
#


class NeuralNetwork():
    # neurons have an ID and a bias, ID is merely the bias' index in genes[0]
    def __init__(self, inputCount:'int', genes:'tuple(list[float], list[str])'):### just make every extra action neuron above what the Environment can handle an automatic, but small, penalty.
        self.connections = list()
        self.internalNeurons = list()
        for i in range(len(genes[0])):# however many bias' there are is how many internal neurons there are
            self.internalNeurons.append(genes[0][i], list())# internal neuron with index as key and contains bias and workingInputs
        


        ### I need to know what number to pick up, how much to mess with it, and where to put it down.
        #                  input or internal     , conection/weight        , and   internal or output?


        other = list()
        lastLayer = list()


        # decode the genome
        
        for i in range(len(genes[1])):### TEST THIS BLOCK rigorously# run through all the genes in the genome
            bitstring = f'{genes[1][i]}'# retrieves the bitstring
            self.connections.append((# disect it
                bitstring[0],
                int(bitstring[1:8], 2),
                bitstring[8],
                int(bitstring[9:16], 2),
                int(bitstring[16:31], 2) / 8000
            ))
            ### put the connections in thinking order.
            if bitstring[8] == '0':# goes to output node##### idk bro, just where I left off.
                lastLayer.append(i)
            else:# goes to internal
                other.append(i)
            ### then store it.
            # so if I'm putting all the connections into a dictionary( so that I can access( without messing with binary) the info. Ya'know, decode.)
            # What info should I use to key them?
            
            # you find out how many connections you have to output nodes, setting them to the side. Then find from the remainder which ones have those connections inputs as their outputs.







            ### source type (input/internal)
            ### source neuron ID modulo source group
            ### output type (internal/action)
            ###


            ### PERCEPTRON, but who goes first? how to keep track? I just need a full set of numbers for each neuron to process...
            # so I have two options, right?
            # the one where each layer is calculated before moving to the next
            # and the way that guy did it... 
            # where instead of the inputs being passed down the layers and transformed along the way...
            # instead, the inputs are just there and the connections are calculated in order.
            # so on init:
            # decode genome
            # get output count, count how many genes specify a connection to an output neuron.
            # put them in a variable, makes it easy to get len.
            # so reversed layer construction? bottom up?
            # then I'd have what exactly?
            # I wouldnt be able to update the inputs mid pass because they might be used by an output neuron directly.




        # We model a single neuron, with 3 input connections and 1 output.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    def __sigmoid(self, x):# You know what the Sigmoid function is... :( not anymore... Sig and Tanh take (-4,4) but Sigm gives (0,1) and Tanh gives (-1,1)
        # The derivative of the Sigmoid function.
        # It indicates how confident we are about the existing weight. The closer to the ends, the less confident.
        '''retuns float in range[0,1]'''
        return 1 / (1 + exp(-x))
    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    def __tanh(self, x, deriv = False):
        '''retuns float in range[-1,1]'''
        if deriv == True:
            return (1 - (tanh(exp2(2) * x)))
          # return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        return tanh(x)
    def __binaryStep(self, x):
        ''' It returns '0' is the input is less then zero otherwise it returns one '''
        return heaviside(x,1)
    def __RELU(self, x):
        ''' It returns zero if the input is less than zero otherwise it returns the given input. '''
        result = []
        for i in x:
            if i < 0:
                result.append(0)
            else:
                result.append(i)
        return result
    


    # # forward pass
    # # # The neural network thinks.
    # def perceptron(self, vectorInput:'list[float]', weights, bias:'float', finalLayer:'bool'):
    #     workingVector = dot(vectorInput, weights.transpose()) + bias
    #     if finalLayer:# activation func
    #         return self.__binaryStep(workingVector)
    #     else:
    #         return self.__tanh(workingVector) 

    def think(self, inputVector):# input shape, list[float]; return shape, list[bool]
        # Pass inputs through our neural network.
        ### I need to calculate the new state of each neuron in order one layer at a time, [0, ...].
        # but the neurons aren't in layers, there's just input/internal/action.
        ### I need to, in order, update the states of the internal neurons so I can produce an output vector from the whole network. [0,0,1,1,0,1,0,1,1,0] or something
        for weight in self.connections:### should be int from string of binary.### should be ordered.
            # find out what you are mutiplying by this weight
            if weight[0]:# is internal neuron input
                workingInput = self.internalNeurons[int(weight[1:8], 2)][1]# get input from list of internal neurons
                # but I'm not multiplying the bias by the weight. I'm multiplying the inputs by the weight... whatever resuly should be stored in the neuron from it's activation func.
            else:# is input neuron
                workingInput = inputVector[int(weight[1:8], 2)]
            ##### x = int(weight[16:31], 2) / 8000
            result = workingInput * x
            # then just add that to the bias of the output node, right? well  dont want to overwrite the bias...
            # so I have to store it so that all the inputs to a node can be added up



            ### PERCEPTRON, but who goes first? how to keep track? I just need a full set of numbers for each neuron to process...
            # so I have two options, right?
            # the one where each layer is calculated before moving to the next
            # and the way that guy did it... 
            # where instead of the inputs being passed down the layers and transformed along the way...
            # instead, the inputs are just there and the connections are calculated in order.
            # so on init:
            # decode genome
            # get output count, count how many genes specify a connection to an output neuron.
            # put them in a variable, makes it easy to get len.
            # so reversed layer construction? bottom up?
            # then I'd have what exactly?
            # I wouldnt be able to update the inputs mid pass because they might be used by an output neuron directly.
            #
            # When you're working with connections instead of nodes, you only have one weight at a time to deal with.
            # for that reason, cant use perceptron node. Must order the connections and something them one at a time.
            # in init, put the internal nodes needed in a list so I can store their bias'.
            
            
            
            
            pass
        return outputVector# 'list[bool]'












##### TRYING TO TAKE INSPIRATION FROM THIS BUT I CANT UNDERSTAND IT
# class Perceptron:
#     def __init__(self, learning_rate, epochs):
#         self.weights = None
#         self.bias = None
#         self.learning_rate = learning_rate
#         self.epochs = epochs

#     def fit(self, X, y):
#         n_features = X.shape[1]
#         self.weights = zeros((n_features))# Initializing weights and bias
#         self.bias = 0
#         for epoch in range(self.epochs):
#             # Traversing through the entire training set
#             for i in range(len(X)):
#                 z = dot(X, self.weights) + self.bias # Finding the dot product and adding the bias
#                 y_pred = self.activation(z) # Passing through an activation function
#                 #Update weights and bias
#                 self.weights = self.weights + self.learning_rate * (y[i] - y_pred[i]) * X[i]
#                 self.bias = self.bias + self.learning_rate * (y[i] - y_pred[i])
#         return self.weights, self.bias

#     def predict(self, X):
#         z = dot(X, self.weights) + self.bias
#         return self.activation(z)





    # BACKUP CODE
    # def think(self, inputVector):
    #     # Pass inputs through our neural network (our single output neuron).
    #     ### I need to calculate the new state of each neuron in order one layer at a time, [0, ...].
    #     for layerIter in self.brain:
    #         for connection in self.brain[layerIter]:### I need to read to connection ()
                # perceptron
    #             tn = self.applyWeight(inputVector)

    #             # Apply bias
    #             ### + connection bias
    #             # Apply activation function
    #         if layerIter == len(self.brain):# Is output layer
    #             result = self.__sigmoid(tn)
    #         else:
    #             result = self.__tanh(tn)
    #     return result



################
    # class SigmoidNeuron:
    #   #intialization
    #   def __init__(self):
    #     self.w = None
    #     self.b = None
    #   #forward pass
    #   def perceptron(self, x):
    #     return dot(x, self.w.T) + self.b

    #   def sigmoid(self, x):
    #     return 1.0/(1.0 + exp(-x))
    #   #updating the gradients using mean squared error loss
    #   def grad_w_mse(self, x, y):
    #     y_pred = self.sigmoid(self.perceptron(x))
    #     return (y_pred - y) * y_pred * (1 - y_pred) * x

    #   def grad_b_mse(self, x, y):
    #     y_pred = self.sigmoid(self.perceptron(x))
    #     return (y_pred - y) * y_pred * (1 - y_pred)
    #   #updating the gradients using cross entropy loss
    #   def grad_w_ce(self, x, y):
    #     y_pred = self.sigmoid(self.perceptron(x))
    #     if y == 0:
    #       return y_pred * x
    #     elif y == 1:
    #       return -1 * (1 - y_pred) * x
    #     else:
    #       raise ValueError("y should be 0 or 1")

    #   def grad_b_ce(self, x, y):
    #     y_pred = self.sigmoid(self.perceptron(x))
    #     if y == 0:
    #       return y_pred
    #     elif y == 1:
    #       return -1 * (1 - y_pred)
    #     else:
    #       raise ValueError("y should be 0 or 1")
    #   #model fit method
    #   def fit(self, X, Y, epochs=1, learning_rate=1, initialise=True, loss_fn="mse", display_loss=False):

    #     # initialise w, b
    #     if initialise:
    #       self.w = random.randn(1, X.shape[1])
    #       self.b = 0

    #     if display_loss:
    #       loss = {}

    #     for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):
    #       dw = 0
    #       db = 0
    #       for x, y in zip(X, Y):
    #         if loss_fn == "mse":
    #           dw += self.grad_w_mse(x, y)
    #           db += self.grad_b_mse(x, y)
    #         elif loss_fn == "ce":
    #           dw += self.grad_w_ce(x, y)
    #           db += self.grad_b_ce(x, y)

    #       m = X.shape[1]
    #       self.w -= learning_rate * dw/m
    #       self.b -= learning_rate * db/m

    #       if display_loss:
    #         Y_pred = self.sigmoid(self.perceptron(X))
    #         if loss_fn == "mse":
    #           loss[i] = mean_squared_error(Y, Y_pred)
    #         elif loss_fn == "ce":
    #           loss[i] = log_loss(Y, Y_pred)

    #     if display_loss:
    #       plt.plot(loss.values())
    #       plt.xlabel('Epochs')
    #       if loss_fn == "mse":
    #         plt.ylabel('Mean Squared Error')
    #       elif loss_fn == "ce":
    #         plt.ylabel('Log Loss')
    #       plt.show()

    #   def predict(self, X):
    #     Y_pred = []
    #     for x in X:
    #       y_pred = self.sigmoid(self.perceptron(x))
    #       Y_pred.append(y_pred)
    #     return array(Y_pred)
###############################
if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    for i in range(10000):
        neural_network.trainWeights(neural_network.think(training_set_inputs), training_set_outputs)

    print(f"New synaptic weights after training: \n{neural_network.synaptic_weights}")

    # Test the neural network with a new situation.

    print(f"Correct answer: \n{training_set_outputs}")
    print(f"Final answer: \n{neural_network.think(training_set_inputs)}")
    print(f"Considering new situation [1, 0, 0] -> ?: {neural_network.think(array([1, 0, 0]))}")
######

# I'm certain we cant fully solve the problem presented in the new Situation with only one neuron

















# from numpy import sqrt


# # DEFINE THE NETWORK

# # Generate random numbers within a bounded normal distribution
# # def truncated_normal(mean=0, sd=1, low=0, upp=10):
# #     return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

# # Create the ‘Nnetwork’ class and define its arguments:
# # Set the number of neurons/nodes for each layer
# # and initialize the weight matrices:
# class Nnetwork:

#     def __init__(self,
#                  no_of_in_nodes,
#                  no_of_out_nodes,
#                  no_of_hidden_nodes,
#                  learning_rate):
#         self.no_of_in_nodes = no_of_in_nodes
#         self.no_of_out_nodes = no_of_out_nodes
#         self.no_of_hidden_nodes = no_of_hidden_nodes
#         self.learning_rate = learning_rate
#         self.create_weight_matrices()

    # def create_weight_matrices(self):
    #     """ A method to initialize the weight matrices of the neural network"""
    #     rad = 1 / sqrt(self.no_of_in_nodes)
    #     X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
    #     self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes, self.no_of_in_nodes))
    #     rad = 1 / sqrt(self.no_of_hidden_nodes)
    #     X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
    #     self.weights_hidden_out = X.rvs((self.no_of_out_nodes, self.no_of_hidden_nodes))

#     def train(self, input_vector, target_vector):
#         pass # More work is needed to train the network

#     def run(self, input_vector):
#         """
#         running the network with an input vector 'input_vector'.
#         'input_vector' can be tuple, list or ndarray
#         """
#         # Turn the input vector into a column vector:
#         input_vector = array(input_vector, ndmin=2).T
#         # activation_function() implements the expit function,
#         # which is an implementation of the sigmoid function:
#         input_hidden = activation_function(self.weights_in_hidden @   input_vector)
#         output_vector = activation_function(self.weights_hidden_out @ input_hidden)
#         return output_vector

# # RUN THE NETWORK AND GET A RESULT

# # Initialize an instance of the class:
# simple_network = Nnetwork(no_of_in_nodes=2,
#                                no_of_out_nodes=2,
#                                no_of_hidden_nodes=4,
#                                learning_rate=0.6)

# # Run simple_network for arrays, lists and tuples with shape (2):
# # and get a result:
# simple_network.run([(3, 4)])









# Just thought:
# put the agent in a client connection for a socket based system
# so the environment runs in the server.
# what does the environment do again?
# cuz the agents recieve observations( in the form of a list of floats)
# and return/send outputs( as a list of bool?)