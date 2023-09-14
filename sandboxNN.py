import struct
from numpy import exp, array, random, dot, exp2, tanh, zeros, heaviside, transpose

def bin_to_float(b):
    """ Convert binary string to a float. """
    bf = int.to_bytes(int(b, 2), 8)  # 8 bytes needed for IEEE 754 binary64.
    return struct.unpack('>d', bf)[0]

def float_to_bin(value):  # For testing.
    """ Convert float to 64-bit binary string. """
    [d] = struct.unpack(">Q", struct.pack(">d", value))
    return '{:064b}'.format(d)





# Genes are in hex string
#


class NeuralNetwork():
    # neurons have an ID and a bias, ID is merely the bias' index in genes[0]
    def __init__(self, inputCount:'int', outputCount:'int', genes:'tuple(list[float], list[str])'):### just make every extra action neuron above what the Environment can handle an automatic, but small, penalty.(Effective blindness, energy cost, ect.)
        self.connections = list()
        self.internalNodes = dict()
        self.outputNodes = list()
        self.layers = list()
        self.nodeLayerStructure = list()# of list of int representing IDs in layers.



        ### The Biasses list includes output and internal neurons. How do I differentiate on the initial list? OutputCout is just the same thing as BiasListLen - internalNeurons, so if I find out how many internal neurons there are some other way...
        #### self.outputNodes = 
        # internal neuron as tuple and contains workingInputs and bias
        # internal neuron as tuple and contains workingInputs and bias.



        for i in range(outputCount):# Build the output nodes.
            self.outputNodes.append((list(), genes[0][i]))# Shape: tuple(int(ID),tuple(list[float], float))
            #                          ID      , inputs, grab the first outputCount biasses leaving the rest for internals.
            pass### is that it? That's all I needed?





        ### I need to know what number to pick up, how much to mess with it, and where to put it down.
        #                  input or internal     , conection/weight        , and   internal or output?


        # decode the genome and build the brain layer by layer( of synapses, not neurons.)
        # We are decoding ALL the genes as weights and storing them in self.connections
        backBurner = list()
        layer = list()
        for i in range(len(genes[1])):### TEST (all of it, but specifically )THIS BLOCK rigorously# run through all the genes in the genome# Decode all the connections into tuples.
            bitstring = f'{genes[1][i]}'# retrieves the bitstring
            decoded = (# disect it
                int(bitstring[0]),# 0 is inputInput, 1 is internal input.
                int(bitstring[1:8], 2),# Specify which node by its ID in that group.
                int(bitstring[8]),# 0 is output, 1 is internal.
                int(bitstring[9:16], 2),# Specify which.
                int(bitstring[16:31], 2) / 8000# the weight.
            )# shape:'tuple(bin, int, bin int, float)
            self.connections.append(decoded)# add it to totalConnections for energy tracking later
            # you find out how many connections you have to output nodes, placing them in the final layer of the Net, setting the rest to the side for now.
            if decoded[2]:# if the output is an internal node
                if any(decoded[3] == id for id in self.internalNodes):# Check if that ID is already in the list.
                    pass# Do nothing
                else:# and if not, init it!## Can't I remove these couple of lines if I change it to "if not any(..."?
                    self.internalNodes[decoded[3]] = (list(), genes[0][i])# Shape: tuple(int(ID),tuple(list[float(weighted results)], float(bias)))# generate node
                backBurner.append(decoded)# then throw it on the...# add connection to location
            else:# goes to output node.
                if decoded[3] < outputCount:# if that int matches a real output ID## I don't need to actually do anything with the specified output node *here*, do I?
                    layer.append(decoded)# it goes in the last layer. Otherwise, pruned.


        if layer.__len__() == 0:# no synapses coupled with the output layer... Brilliant.
            ### possibly find a way to have *this* version(self) generate a just an empty output vector always. nahhh maybe...
            ### make sure the think function is alright with handling the output nodes at the very least, and without inputs if needed.
            pass
        self.layers.append(layer)
        if len(backBurner) >= 1:# That's not the whole brain, there's another layer.
            nonFinalLayerConstruction = True
        else:
            nonFinalLayerConstruction = False
        workingIter = 0
        lastLayerInputIDs = list()
        ### populate lastLayerInputIDs
        ##### I STG I was trying to do something important +- 10 lines of here...
        ### Then find from the remainder which ones have those connections' input nodes as their outputs. Setting them as the next from last layer. Repeat until...? No connections can be made for a layer.
        while nonFinalLayerConstruction:
            layer = list()
            layerInputIDs = list()
            for connection in self.layers[workingIter]:# from the previous layer of connections...# If it's not on the list,add it.
                if not any(f"{connection[0]}:{connection[1]}" == x for x in layerInputIDs):# copy only the unique IDs
                    layerInputIDs.append(f"{connection[0]}:{connection[1]}")# ...read out all the input locations.( to a list.)
                    if any(f"{connection[0]}:{connection[1]}" == x for x in lastLayerInputIDs):# if current connection input ID is also an input in the previous layer, kill it from the list and re-add it.
                        del self.nodeLayerStructure[workingIter][connection[1]]### list func to remove specifically the node ID from the previous layer of nodeLayerStructure.
                if connection[0]:# is internal input
                    self.nodeLayerStructure[workingIter + 1].append(connection[1])
            for connection in backBurner:# for every connection in the back burner# Make the next layer
                if any(f"{connection[2]}:{connection[3]}" == x for x in lastLayerInputIDs):# if the connections' output matches the input from a connection in the previous layer:
                    layer.append(connection)### it is part of the next layer.
                    pass## Pretty sure that's all I need here...
            if not layer:# Layer was empty
                nonFinalLayerConstruction = False# Brain finished
            else:
                self.layers.append(layer)
            lastLayerInputIDs = layerInputIDs
            workingIter += 1
        self.layers.reverse()
        self.nodeLayerStructure.reverse()

        # for layer in self.layers:
        #     for connection in layer:
        #         if connection




        # so on init:
        # decode genome
        # get output count, count how many genes specify a connection to an output neuron.
        #
        # so reversed layer construction? bottom up?
        # then I'd have what exactly?
        # I wouldnt be able to update the inputs mid pass because they might be used by an output neuron directly.




        # # We model a single neuron, with 3 input connections and 1 output.
        # # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # # and mean 0.
        # self.synaptic_weights = 2 * random.random((3, 1)) - 1

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



    # forward pass # The neural network thinks.
    def think(self, inputVector:'list[float]'):
        '''return shape, list[bool]'''

        ### Define default output vector.
        outputVector = list()
        ### I need to produce an output vector from the whole network. Bool or Binary. Alternitavely, float(0,1).






        ### make sure the think function is alright with handling the output nodes at the very least, and without inputs if needed.

        # Pass inputs through our neural network.
        ### I need to calculate the new state of each neuron in order one layer at a time, [0, ...].
        # but the neurons aren't in layers, there's just input/internal/action.
        if not self.layers[0]:# only possible activation is if it's the first layer. Therefore, empty brain.
            for node in self.outputNodes:
                outputVector.append(self.__binaryStep(0 + node[1]))
        else:# not empty brain.
            for layer in self.layers:### should be int from string of binary.### should be ordered.# Cycle through all of the
                for synapse in layer:# find out what you are mutiplying by this weight: Working Input.
                    if synapse[0]:# if input's coming from an internal neuron
                        workingInput = self.internalNodes[synapse[1]]# get working input from which one.
                    else:# if input's coming from an input neuron
                        workingInput = inputVector[synapse[1]]# get working input from which one.
                    result = workingInput * synapse[4]# Now that we found the working input, we multiply it by the weight of the current synapse
                    ### read line above.

                    # then just add that to the bias of the output node, right? well  dont want to overwrite the bias...
                    # so I have to store it so that all the inputs to a node can be added up
                    
                    
                    # store it in the cache of the propper node.

                    if synapse[2]:# is internal output
                        ### store the result.
                        self.internalNodes[synapse[3]][0].append(result)
                    else:# is output neuron
                        self.outputNodes[synapse[3]][0].append(result)# synapse[oddNum] returns an ID
                        pass

                                                                                                                                                # bitstring[0],# 0 is inputInput, 1 is internal input.
                                                                                                                                                # int(bitstring[1:8], 2),# Specify which node by its ID in that group.
                                                                                                                                                # bitstring[8],# 0 is output, 1 is internal.
                                                                                                                                                # int(bitstring[9:16], 2),# Specify which.
                                                                                                                                                # int(bitstring[16:31], 2) / 8000# the weight.
                ### That neuron needs to fire before the next layer can go.
                # maybe... if connection input is internal, then I know a neuron needs to have fired before current connection can calculate.

            #----------- THAT DOESNT MAKE ANY SENSE, BRO.
            # ### PERCEPTRON, but who goes first? how to keep track? I just need a full set of numbers for each neuron to process...
            # # so I have two options, right?
            # # the one where each layer is calculated before moving to the next
            # # and the way that guy did it... 
            # # where instead of the inputs being passed down the layers and transformed along the way...
            # # instead, the inputs are just there and the connections are calculated in order.
            # # so on init:
            #-----------
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
        return outputVector










# def perceptron(self, vectorInput:'list[float]', weights, bias:'float', outputToAction:'bool'):
#     workingVector = dot(vectorInput, weights.transpose()) + bias
#     if outputToAction:# determines activation func
#         return self.__binaryStep(workingVector)
#         # return self.__RELU(self.__sigmoid(workingVector))## if you want float, instead of bool output,
#     else:
#         return self.__tanh(workingVector)


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



# # # # # # BACKUP CODE

# def int_to_bytes(n, length):  # Helper function
#     """ Int/long to byte string.

#         Python 3.2+ has a built-in int.to_bytes() method that could be used
#         instead, but the following works in earlier versions including 2.x.
#     """
#     return codecs.decode('%%0%dx' % (length << 1) % n, 'hex')[-length:]


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

# I'm confident we cant fully solve the problem presented in the new Situation with only one neuron

















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