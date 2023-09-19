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

    def proccessFinalNodeLayer(self):
        '''return outputVector\n
        as a 1-D vector of returns from the binaryStep activation function.'''
        outputVector = list()
        for currentNode in self.outputNodes:# make sure to use the right activation function
            outputVector.append(self.__binaryStep(perceptron(currentNode, self.__sigmoid) - 0.92))## have this threshold start low(even so far as 0) and increase infinitely ever closer to 0.95(or maybe even .98, but I wouldnt go higher...) as generation count increases.
        return outputVector

    def proccessInternalNodeLayer(self, layer:'list[int]'):# for node in range nodeLayerInUse
        #generate list of float(-1,1) aka result of tanh
        '''Updates internal Nodes' states in place.\nreturn None'''
        for i in range(len(layer)):# for every Node in the provided layer of IDs, set the output state of that Node while also resetting it's input cache.
            self.internalNeurons[layer[i]] = perceptron(self.internalNeurons[layer[i]], self.__tanh)

    def __init__(self, outputCount:'int', genes:'tuple(list[float], list[str])'):### just make every extra action neuron returned above what the Environment can handle an automatic, but small, penalty.(Blindness, energy cost, ect.)
        self.cost = (int(), float())## print this once, I just wanna see it.
        self.outputNodes = list()# Shape: list(tuple(list[float(results from weight multiplications)], float(node bias)). key : list[tuple(inputs, bias)]
        self.internalNeurons = dict()# internal neuron is tuple and contains workingInputs(list[float(-4,4)]), bias(float(-4,4)), and float(-1,1). key : dict{ID:tuple(inputs, bias, state)}.
        self.layersSynapse = list()# Full decoded Connections, stacked in layers.# Shape: list[list[tuple(int, int, int, int, float)]]
        self.layersNeuron = list()# Connection output IDs(internal and output Neurons) stacked in layers.# Shape: list[list[outID]]### standardize this out.
        self.layersNeuron.append(list())# create the first empty layer of Node struct.
        for i in range(outputCount):# add all output nodes to the first layer of the Neuron structure
            self.outputNodes.append((list(), genes[0][i], float()))# Shape: tuple(tuple(list[float(input storage)], float(bias), float(output storage))
            self.layersNeuron[0].append(i)# List of ID pointers. That's it. Because all output Neurons are in the last layer, they don't need output type.
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
            )
            self.cost.append(decodedSynapse)# add it to totalConnections for energy tracking later
            # you find out how many connections you have to output nodes, placing those in the final layer of the synapseStructure, setting the rest to the side for now.
            if decodedSynapse[2]:# if the output is an internal node, just go ahead and add that to the last layer, first.
                if not any(decodedSynapse[3] == id for id in self.internalNeurons):# Check if that ID is already in the list.
                    self.internalNeurons[decodedSynapse[3]] = (list(), genes[0][outputCount + len(self.internalNeurons)], float())# generate internal node with correct bias
                backBurner.append(decodedSynapse)# then throw it on the...
            else:# Connects to an output node.
                if decodedSynapse[3] < outputCount:# is a real output Node. if not real, pruned.
                    workingLayerSynapses.append(decodedSynapse)# it goes in the last layer.
                    thisLayerInputIDs.append((decodedSynapse[0], decodedSynapse[1]))# log input for use in finding the next Connection and Node layers.
        # First( base) layer of Synapses.
        self.layersSynapse.append(workingLayerSynapses)# take the layer[every Connection that goes to an output Node] and Add it to allSynapseLayers first.
        needToBuildNextLayer = True
        if not len(backBurner) >= 1:# ran out of Connections to make layers with.
            needToBuildNextLayer = False
        elif not len(workingLayerSynapses):# no synapses coupled with the previous Node layer.
            needToBuildNextLayer = False
        while needToBuildNextLayer:# try to build a new connection layer
            lastLayerInputIDs = thisLayerInputIDs
            thisLayerInputIDs = list()
            self.layersNeuron.append(list())
            # Build Node layer
            for determinedConnection in self.layersSynapse[len(self.layersNeuron) - 1]:# from the previous layer of connections...# If it's not on the list,add it.
                if not any((determinedConnection[0], determinedConnection[1]) == x for x in thisLayerInputIDs):# copy only the unique IDs. No duplicate computations of Nodes.
                    thisLayerInputIDs.append((determinedConnection[0], determinedConnection[1]))# add it to the list of input locations. Ie, next Node layer.
                    if any((determinedConnection[0], determinedConnection[1]) == x for x in lastLayerInputIDs):# if current input ID matches an input ID in the previous layer, promote it.
                        self.layersNeuron[len(self.layersNeuron) - 1].remove((determinedConnection[0], determinedConnection[1]))# remove the node ID from the previous layer of self.nodeLayerStructure
                if determinedConnection[0]:#if connection has input from an internal neuron(, as opposed to input from input vector).
                    self.layersNeuron[len(self.layersNeuron)].append(determinedConnection[1])# currentLayer.append(internalNodeID)
            # Build connection layer
            workingLayerSynapses = list()
            for connection in backBurner:# for every connection in the back burner# Make the next layer
                # the drop off location can't be an output node. because on the first run everything with an output Node as their output will already be put away.
                if any((connection[2], connection[3]) == id for id in lastLayerInputIDs):# if the connections' output matches an input ID from a connection in the previous layer:
                    workingLayerSynapses.append(connection)# it is part of the next layer.
                    ### anything else?
            if workingLayerSynapses:# if there was something there, add it to the stack.
                self.layersSynapse.append(workingLayerSynapses)
            else:# Layer was empty
                needToBuildNextLayer = False# Brain finished, stop building. Last layer was the top layer
            if 

            #####
            # if not len(backBurner) >= 1:# ran out of Connections to make layers with.
            #     needToBuildNextLayer = False
            # elif not len(workingLayerSynapses):# no synapses coupled with the previous Node layer.
            #     needToBuildNextLayer = False
            # in one run of this code up to this point should leave us with
            # the layer X1
            # Layer Y1
            # loop: i += 1
                # Layer Xi+1
                # Layer Yi+1
            # so, it stops when it either:
            #     runs out of connections
            #     or ends the loop with an empty layer of connections
            


            # build base layer nodes
            # find next Node layer from prevConnectionLayer inputs, adding it to nodeStruct.
            # cycle through remaining connections collecting those whos output IDs are in list of last layer input IDs.
            # 


            # build output layer from outputCount and genes[bias]
            # decode genome to get all Connections
            # init all internal Nodes from output IDs of decoded Connections
            # build base connection layer from output layer and internal nodes
                # if no connections left or no connections output to outputs: brain done
            # loop
                # get input IDs from previous layer of Connections
                # build next internal Neuron layer from input IDs and internal Nodes
                # build next Connection layer from input IDs and remaining Connections
                    # if no connections left to look at or no connections output to prevLayerInputs: brain done



            # 



        self.layersSynapse.reverse()
        self.layersNeuron.reverse()### I have to make self.outputNodes accesible to self.think() somehow...
        

        # for layer in self.layers:
        #     for connection in layer:# cycle through all the connections in self.brain
        #         if connection






        # # We model a single neuron, with 3 input connections and 1 output.
        # # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # # and mean 0.
        # self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # forward pass # The neural network thinks.
    def think(self, inputVector:'list[float]'):# Prepare input set for our neural network.
        '''return list[single-bit binary]'''# I need to produce an output vector from the whole network. List of Bool or Binary. IDK
        # I need to calculate the state of each neuron in order one layer at a time, self.[0: -1].
        thinkingLayerNeurons = int()# If I need to run nodes first, do so.
        if len(self.layersNeuron) - 1 == len(self.layersSynapse):# if equal, connections goes first like normal. If nodeLayerCount >1 connectionLayerCount, run the node layer first. otherwise, something broke...
            self.proccessInternalNodeLayer(self.layersNeuron[thinkingLayerNeurons])# run the first node layer before starting the connection computations
            thinkingLayerNeurons += 1
        for countLayerSynapse in self.layersSynapse:# go back and forth between running a connection layer and a node layer.
            for synapse in self.layersSynapse[countLayerSynapse]:# find out what you are mutiplying by this weight: Working Input.
                if synapse[0]:# if input's coming from an internal neuron
                    result = mult(synapse[4], self.internalNeurons[synapse[1]])# get working input from which one.
                else:# if input's coming from a raw input
                    result = mult(synapse[4], inputVector[synapse[1]])# get working input from which one.
                if synapse[2]:# if output is internal# store result in the cache of the propper node.
                    self.internalNeurons[synapse[3]][0].append(result)# referencing: (suposedly )empty list
                else:# is output neuron
                    self.outputNodes[synapse[3]][0].append(result)# synapse[oddNum] returns an ID.
            # until the last node layer which needs to be skipped
            if not countLayerSynapse == len(self.layersSynapse):# When I am calculating the last synapse layer, I know the next Node layer is the last one.
                self.proccessInternalNodeLayer(self.layersNeuron[thinkingLayerNeurons])# After every synapse in the layer has been calculated, The next layer of neurons needs to fire.
                thinkingLayerNeurons += 1# gather input values and proccess the layer of Nodes with each advancement.
        # run final output node layer
        return self.proccessFinalNodeLayer(self)


# # # # # # BACKUP CODE
###############################################
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




# We are decoding ALL the genes as weights and storing them in layers based on their I/O targets.






# # so I have two options, right?
# # the one where each layer is calculated before moving to the next
# # and the way that guy did it...
# # where instead of the inputs being passed down the layers and transformed along the way...
# # instead, the inputs are just there and the connections are calculated in order.
# I think I've created a combination of the two concepts
# if so, I've surely done it by structuring the network with all the outputs neurons in the last lasyer, first.
