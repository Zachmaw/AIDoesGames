# the comment key I always go by
# (hopefully )Informative comment.
## something to come back to when the project is complete.
### TO-DO
#### probably an accident.
##### Bookmark. I had to leave off here.
# yep... Thaaat's my system...

from numpy import exp, array, random, dot, exp2, tanh, zeros, heaviside, transpose

# nothing calls these yet. Presumably, Genome class will.
# def bin_to_float(b):
#     """ Convert binary string to a float. """
#     bf = int.to_bytes(int(b, 2), 8)  # 8 bytes needed for IEEE 754 binary64.
#     return struct.unpack('>d', bf)[0]
# def float_to_bin(value):  # For testing.
#     """ Convert float to 64-bit binary string. """
#     [d] = struct.unpack(">Q", struct.pack(">d", value))
#     return '{:064b}'.format(d)

# Genes are stored in hex string format. mutations should happen before encoding.# I dissagree, Mutations happen either during encoding or decoding.
# but the simulation should handle mutations?




        ### where tf does this go?!    # you find out how many connections you have to output nodes, placing them in the final layer of the Net, setting the rest to the side

            ### possibly find a way to have *this* version(self) generate a just an empty output vector always. nahhh. bad idea. trust.
            ### make sure the think function is alright with handling the output nodes at the very least, and without inputs if needed.



def binToHex(n):
    return hex(num = int(n, 2))

def mult(weightValue:'float', input):
    '''return float'''
    ### if BIAS == 'synapse':
        # you get the idea?
    return input * weightValue

def perceptron(node:'tuple(list[float], float, float)', activationFuncOfChoice):# AKA, it shouldn't have an ID by this point. No lookups.
    tempN = float()# add up the inputs which are already stored
    for flo in node[0]:
        tempN += flo
    return (list(), node[1], activationFuncOfChoice(tempN + node[1]))# add bias# Activate.### if list() doesn't work here, you can just swap it for node[0].

class NeuralNetwork():
    # For Internal Nodes, ID is the node's ID as an int key in self.internalNodes:'dict'
    # For Output Nodes, ID is merely the node's index in self.outputNodes:'list'
    
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

    def finalNodeLayer(self):
        '''return outputVector'''
        outputVector = list()
        for currentNode in self.outputNodes:### for node in range nodeLayerInUse# make sure to use the right activation function
            outputVector.append(self.__binaryStep(perceptron(currentNode, self.__sigmoid) - 0.92))## have this threshold start low(even so far as 0) and increase infinitely ever closer to 0.95 as generation count increases.
        return outputVector

    def __init__(self, inputCount:'int', outputCount:'int', genes:'tuple(list[float], list[str])'):### just make every extra action neuron returned above what the Environment can handle an automatic, but small, penalty.(Blindness, energy cost, ect.)
        self.cost = (int(), float())## print this once, I just wanna see it.
        # self.hasNodeOverConn = False# if len(self.nodeLayerStructure) == len(self.layersSynapse):?
        self.outputNodes = list()# Shape: list(int(ID), tuple(list[float(results from weight multiplications)], float(node bias)))
        self.internalNeurons = dict()# internal neuron is tuple and contains workingInputs(list[float(-4,4)]), bias(float(-4,4)), and float(-1,1). key : tuple(list, bias, state).
        self.layersSynapse = list()# These three(?) lists together represent the completed brain.# Shape: list[], list[]
        self.layersNeuron = list()# of list lists of tuple of ints representing IDs in layers. list[list[tuple(outType, outID)]]?

        for i in range(outputCount):# Build the output nodes.
            self.outputNodes.append((list(), genes[0][i], float()))# Shape: tuple(tuple(list[float(input storage)], float(bias), float(output storage))
            #                        inputs, grab the first {outputCount} biasses leaving the rest for internals.
            pass### What am I missing?

        # decode the genome and build the brain layer by layer, from the bottom up.
        # We are decoding ALL the genes as weights and storing them in layers based on their I/O targets.
        backBurner = list()
        workingLayerSynapses = list()
        thisLayerInputIDs = list()
        for i in range(len(genes[1])):# run through all the genes in the genome# Decode all the connections into tuples.# for synapse in the_genome:
            bitstring = f'{genes[1][i]}'# retrieves the bitstring### is that how you retrieve a bitstring from hex? f'{0xABC123EFFF:0>42b}'
            decodedSynapse = (# disect it
                int(bitstring[0]),# 0 is inputInput, 1 is internal input.
                int(bitstring[1:8], 2),# Specify which node by its ID in that group.
                int(bitstring[8]),# 0 is output, 1 is internal.
                int(bitstring[9:16], 2),# Specify which.
                int(bitstring[16:31], 2) / 8000# the weight.
            )# shape:'tuple(bin, int, bin int, float)
            self.cost.append(decodedSynapse)# add it to totalConnections for energy tracking later
            # you find out how many connections you have to output nodes, placing them in the final layer of the Net, setting the rest to the side for now.
            if decodedSynapse[2]:# if the output is an internal node, just go ahead and add that to the last layer, first.
                if any(decodedSynapse[3] == id for id in self.internalNeurons):# Check if that ID is already in the list.
                    pass# Do nothing. Can't use it yet. The connection OR the node. Besides, that Node has already been assigned a Bias.
                else:# and if it doesn't exist yet, create it!## Can't I remove these two lines if I change it to "if not any(..."?
                    self.internalNeurons[decodedSynapse[3]] = (list(), genes[0][outputCount + len(self.internalNeurons)], float())# generate internal node with correct bias
                backBurner.append(decodedSynapse)# then throw it on the...
            else:# goes to output node.# the node already exists.
                if decodedSynapse[3] < outputCount:# is a real output Node. if not real, prune.
                    workingLayerSynapses.append(decodedSynapse)# it goes in the last layer.
                    thisLayerInputIDs.append((decodedSynapse[0], decodedSynapse[1]))# log input



        # First( last) layer of Neurons and Synapses.
        self.layersSynapse.append(workingLayerSynapses)# take the layer[every Connection that goes to an output Node] and Add it to allSynapseLayers first.### When I run out of Node layers and am calculation the last synapse layer, I know the next Node layer is the last one.

        self.layersNeuron.append(list())# add all output nodes to the first layer of the Neuron structure
        for nodeID in range(len(self.outputNodes)):
            self.layersNeuron[0].append(nodeID)# List of ID pointers. That's it.

        ### I know this bit is ugly code. Help me, don't mock me.
        if len(backBurner) >= 1:# That's not the whole brain, there's more synapses waiting.
            if not len(workingLayerSynapses):# no synapses coupled with the first Node layer.
                # surely not all output nodes are always going to have connections, just as surely not all connections will be used every time.
                needToBuildNextLayer = False
            else:
                needToBuildNextLayer = True
        else:# ran out of Connections to make layers with.
            needToBuildNextLayer = False


            # what we need is to 
            # for connection

                # init all internal nodes# done
                # build base layer nodes# done
                # build base layer connections# done
                # find next Node layer from prevConnectionLayer inputs, adding it to nodeStruct.
                # cycle through remaining connections collecting those whos output IDs are in list of last layer input IDs.
                # 


        ### populate lastLayerInputIDs
        ### Something to do with activation functions.
        ### Then find from the remainder which ones have those connections' input nodes as their outputs. Setting them as the next from last layer. Repeat until...? No connections can be made for a layer.
        while needToBuildNextLayer:# try to build a new connection layer
            lastLayerInputIDs = thisLayerInputIDs
            thisLayerInputIDs = list()
            workingLayerSynapses = list()
            self.layersNeuron.append(list())
            # Build Node layer
            for determinedConnection in self.layersSynapse[len(self.layersNeuron) - 1]:# from the previous layer of connections...# If it's not on the list,add it.
                if not any((determinedConnection[0], determinedConnection[1]) == x for x in thisLayerInputIDs):# copy only the unique IDs. No duplicate computations of Nodes.
                    thisLayerInputIDs.append((determinedConnection[0], determinedConnection[1]))# add it to the list of input locations. Ie, next Node layer.
                    if any((determinedConnection[0], determinedConnection[1]) == x for x in lastLayerInputIDs):# if current input ID matches an input ID in the previous layer, promote it.
                        self.layersNeuron[len(self.layersNeuron) - 1].remove((determinedConnection[0], determinedConnection[1]))# remove the node ID from the previous layer of self.nodeLayerStructure
                if determinedConnection[0]:#if connection has input from an internal neuron(, as opposed to input from input vector).
                    self.layersNeuron[len(self.layersNeuron)].append(determinedConnection[1])# currentLayer.append(internalNodeID)
                # else:# input vector
                #     pass### connections need to be able to pull data from the input vector
            # Build connection layer
            for connection in backBurner:# for every connection in the back burner# Make the next layer
                ### the drop off location must be internal node. Huh? why? Output nodes can hold incoming data too.
                if any((connection[2], connection[3]) == id for id in lastLayerInputIDs):# if the connections' output matches an input ID from a connection in the previous layer:
                    workingLayerSynapses.append(connection)### it is part of the next layer.
                    pass
            if workingLayerSynapses:# if there was something there, add it to the stack.
                self.layersSynapse.append(workingLayerSynapses)
            else:# Layer was empty
                needToBuildNextLayer = False# Brain finished, stop building. Last layer was the top layer### weights that sample from internal nodes take the bias?
            # in one run of this code up to this point should leave us with
            # the output Node layer
            # the base internal node layer
            # and the 2 bottom connection layers
            # meaning the next thing to find is node layer
            # but it's already handled by the loop?
            # so, it stops when it either:
            #     runs out of connections
            # or ends the loop with an empty layer of connections
        self.layersSynapse.reverse()
        self.layersNeuron.reverse()### I have to make self.outputNodes accesible to self.think() somehow...
        

        # for layer in self.layers:
        #     for connection in layer:# cycle through all the connections in self.brain
        #         if connection






        # # We model a single neuron, with 3 input connections and 1 output.
        # # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # # and mean 0.
        # self.synaptic_weights = 2 * random.random((3, 1)) - 1

    def internalNodeLayer(self, layer:'list[int]'):
        #generate list of float(-1,1) aka result of tanh
        '''Updates internal Nodes' states in place.\nreturn None'''
        for i in range(len(layer)):
            self.internalNeurons[layer[i]] = perceptron(self.internalNeurons[layer[i]], self.__tanh)


    # forward pass # The neural network thinks.
    def think(self, inputVector:'list[float]'):
        '''return list[bool]'''







        ### Define default output vector.
        outputVector = list()### I need to produce an output vector from the whole network. List of Bool or Binary.






        ### make sure the think function is alright with handling the output nodes at the very least, and without inputs if needed.

        # Pass inputs through our neural network.
        ### I need to calculate the new state of each neuron in order one layer at a time, [0, ...].
        # If I need to run nodes first, do so.
        # then go back and forth between running a connection layer and a node layer.
        
        if not self.layersSynapse[0]:# only possible activation is if the first layer contains nothing. Therefore, it's the only layer and the brain is empty.
            # run the outputNodeLayer and return the resulting vector
            for node in self.outputNodes:
                outputVector.append(self.__binaryStep(0 + node[1]))### wrote a func
        else:# not empty brain.
            ### if self.hasNodeOverConn:# bool
            if len(self.layersNeuron) + 1 == len(self.layersSynapse):# if equal, connections goes first. If node >1 connection, node goes first. otherwise, it broke...
                self.internalNodeLayer(self.layersNeuron[0])
                pass
                # run the first node layer before starting the connection computations
            # get top layer






            for layer in self.layersSynapse:### should be int from string of binary.### should be ordered.# Cycle through all of the
                # Proccess current Connection layer.
                for synapse in layer:# find out what you are mutiplying by this weight: Working Input.
                    # do I want to pass in IDs?
                    ### no, I want to JUST pass in the weight float and the propperly identified input.
                    if synapse[0]:# if input's coming from an internal neuron
                        result = mult(synapse[4], self.internalNeurons[synapse[1]])# get working input from which one.
                    else:# if input's coming from an input neuron
                        result = mult(synapse[4], inputVector[synapse[1]])# get working input from which one.
                    if synapse[2]:# if output is internal# store result in the cache of the propper node.
                        self.internalNeurons[synapse[3]][0].append(result)# referencing: (suposedly )empty list
                    else:# is output neuron
                        self.outputNodes[synapse[3]][0].append(result)# synapse[oddNum] returns an ID.

                                                                                                                                                # bitstring[0],# 0 is inputInput, 1 is internal input.
                                                                                                                                                # int(bitstring[1:8], 2),# Specify which node by its ID in that group.
                                                                                                                                                # bitstring[8],# 0 is output, 1 is internal.
                                                                                                                                                # int(bitstring[9:16], 2),# Specify which.
                                                                                                                                                # int(bitstring[16:31], 2) / 8000# the weight.
                # After every synapse in the layer has been calculated,
                # The next layer of neurons needs to fire before the next layer of synapses can go.
                self.internalNodeLayer(self.layersNeuron[])###### proccess current node layer


                # Can't I just determine whether node layer needs to run first by the counts of connLayer and nodeLayer?

                # if not len(self.layersSynapse):### if no layers in synapse layers???
                #     pass
            ### A layer of Nodes needs to produce an output vector for the next weight layer to compute.





            # # so I have two options, right?
            # # the one where each layer is calculated before moving to the next
            # # and the way that guy did it...
            # # where instead of the inputs being passed down the layers and transformed along the way...
            # # instead, the inputs are just there and the connections are calculated in order.
            # I think I've created a combination of the two concepts
            # if so, I've surely done it by structuring the network with all the outputs neurons in the last lasyer, first.


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