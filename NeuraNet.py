#MISSION STATEMENT
# IMPORTS
from numpy import random, exp, exp2, tanh, heaviside, array#, dot, transpose
from math import ceil
# import timeit
# CONSTANT INIT
HEX_OPTIONS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F"]
Biases = list()
for i in range(16):# could only get -8 to +7, but that's fine because a bias of +8 means that neuron fires(with max power) no matter what.
    Biases.append(int(i - 8))
# SETTINGS INIT
# FUNCTIONS
def mult(weightValue:'float', input:"float"):
    '''return float'''
    return input * weightValue

def binToHex(binaryString):
    return hex(int(binaryString, 2))

def hextobin(hexaString):
  return bin(int(hexaString, 16))[2:].zfill(len(hexaString) * 4)

def replace_str_index(text, index=0, replacement=''):
    return f'{text[:index]}{replacement}{text[index+1:]}'

def roll(d, dc, bonus):
    r = random.randint(1, d+1)
    if r == d:# if critical roll
        return True
    if r == 1:# if natural 1
        return False
    if dc <= r + bonus:
        return True
    else:
        return False

def randomOneGene():
    gene = list()
    for i in range(9):
        gene.append(random.choice(HEX_OPTIONS))
    return "".join(gene)

def init_random_Genome(geneCount:"int"):
    genome = list()
    for i in range(geneCount):
        genome.append(randomOneGene())
    return genome# a list of hexdec strings( each with len(9))

def generateMutationBitstring(geneLen:"int", toxicWasteBonus:"float"=0.0):
    '''Default odds: 8/1000\nA float of 1.0 should cause mutation chance to be 999/1000\n Nat 1 is still a possibility.'''
    temp = list()
    for i in range(geneLen):
        if roll(1000, 993, 1000 * toxicWasteBonus):
            temp.append("1")
        else:
            temp.append("0")
    return "".join(temp)

def bitCombine(argA:"str", argB:"str"):
    temp = list()
    for argi in range(len(argA)):
        temp.append(str((int(argA[argi]) + int(argB[argi])) % 2))
    return "".join(temp)

def mutateBitstring(bitstring:"str"):
    return bitCombine(bitstring, generateMutationBitstring(36))

def mutateHexdec(gene:"str", radiationBonus:"float"):
    '''raises/lowers the value of random bonds by one'''
    for i in range(len(gene)):
        if roll(1000, 994, 1000 * radiationBonus):
            gene = replace_str_index(gene, i, HEX_OPTIONS[int(hextobin(gene[i]), 2) + random.choice([1, 15]) % 16])
    return gene

def perceptron(node:'tuple(list[float], float, float)', activationFuncOfChoice):# AKA, it shouldn't have an ID by this point. No lookups.
    tempN = float()# add up the inputs which are already stored
    for floa in node[0]:
        tempN += floa
    result = activationFuncOfChoice(tempN + node[1])# the result of a node is a float no matter what, internal(-1,1), output(0,1). But then we binary the output nodes. 
    return (list(), node[1], result)# add bias# Activate.

# CLASSES
class NeuralNetwork():
    # For Internal Nodes, ID is the node's ID as an int key in self.internalNodes:'dict'
    # For Output Nodes, ID is merely the node's index in self.outputNodes:'list'
    def __init__(self, outputCount:'int', generation:"int", genes:'list[str]'=None, irradiation:"float"=0.01):
        ### self.costToExist = (int(), float())# cost to exist per turn
        if genes == None:# make some genes based on generation
            genes = init_random_Genome(ceil(generation/10))## might lower this number, likely no further than 7.
        self.outputNodes = list()# Shape: list(tuple(list[float(results from weight multiplications)], float(node bias)). key : list[bias,inputs]
        self.internalNeurons = dict()# internal neuron is tuple and contains workingInputs(list[float(-4,4)]), bias(int(-4,3)), and output(float(-1,1)). key : dict{ID:tuple(list[bias,inputs], state)}.
        self.layersSynapse = list()# Full decoded Connections, stacked in layers.# Shape: list[list[tuple(int, int, int, int, float)]]
        self.layersNeuron = list()# Connection output IDs(internal and output Neurons), stacked in layers.(Bro, trust me.)# Shape: list[list[outID]]
        self.layersNeuron.append(list())# create the first empty layer of Node struct.
        for i in range(outputCount):# add all output nodes to the first layer of the Neuron structure
            self.outputNodes.append((list(), int()))# Shape: tuple(tuple(list[float(input storage)], float(bias), float(output storage))
            self.layersNeuron[0].append(i)# List of ID pointers. That's it. Because all output Neurons are in the last layer, they don't need output type.
        backBurner = list()
        workingLayerSynapses = list()
        currentLayerInputIDs = list()
        genome = list# for every gene, make it a bitstring, mutate it, then disect it, turn it back into hexString, and store it in genome.
        for i in range(len(genes)):# run through all the genes in the genome# Decode all the connections into tuples.# for synapse(index) in the_genome:
            bitstring = mutateBitstring(hextobin(mutateHexdec(genes[i], irradiation)))# MUTATION * 2 COMBO | By way of irradiation and for growing NNs.
            genome.append(binToHex(bitstring))
            decodedSynapse = (# disect it 
                int(bitstring[0]),# 0 is inputInput, 1 is internal input. 1 bit
                int(bitstring[1:8], 2),# Specify which node by its ID in that group. 7 bits
                int(bitstring[8]),# 0 is output, 1 is internal. 1 bit
                int(bitstring[9:16], 2),# Specify which. 7 bits
                int(bitstring[16:32], 2) / 8000,# the weight. 16 bits
                int(bitstring[32:], 2)# Bias as int representing an index in the list of preset biases. # 4 bits, 0 to 15, -8 to +7.
            )# We are decoding ALL the genes as weights and storing them in layers based on their I/O targets.
            # you find out how many connections you have to output nodes, placing those in the final/first layer( you know what I mean) of the synapseStructure, setting the rest to the side for now.
            if decodedSynapse[2]:# if the output( from this gene) is an internal node, (make it.)
                if not any(decodedSynapse[3] == id for id in self.internalNeurons.keys()):# Check if that ID is not already in the dict.
                    self.internalNeurons[decodedSynapse[3]] = (list(), Biases[decodedSynapse[5]], float())# generate internal node with correct bias
                backBurner.append(decodedSynapse)# then throw it on the...!
            else:# Connects to an output node. just add that to the last layer, first.
                if decodedSynapse[3] <= outputCount:# is a real output Node. if selected ID is invalid, prune.
                    workingLayerSynapses.append(decodedSynapse)# it goes in the last layer.
                    currentLayerInputIDs.append((decodedSynapse[0], decodedSynapse[1]))# log input for use in finding the next Connection and Node layers.
                    if decodedSynapse[0]:# if it's internal, create it so it *can* be found.
                        if not any(decodedSynapse[1] == id for id in self.internalNeurons.keys()):# Check if that ID is not already in the dict.
                            self.internalNeurons[decodedSynapse[1]] = (list(), Biases[decodedSynapse[5]], float())# generate internal node with correct bias
        self.genome = genome
        # First( base) layer of Synapses( which all output to output Nodes).
        self.layersSynapse.append(workingLayerSynapses)# take the layer[every Connection that goes to an output Node] and Add it to allSynapseLayers first.
        needToBuildNextLayer = True
        if not len(backBurner) >= 1:# ran out of Connections to make layers with.
            needToBuildNextLayer = False
        elif not len(workingLayerSynapses):# no synapses coupled with the previous Node layer.
            needToBuildNextLayer = False
        while needToBuildNextLayer:# try to build a new set of layers[1:]
            lastLayerInputIDs = currentLayerInputIDs
            currentLayerInputIDs = list()
            self.layersNeuron.append(list())
            # Build Node layer## I feel like somethins fucky with lastLayerInputIDs because thisLayerInputIDs is always being poulated by inputIDs from determined connections.
            #                                              yeah, so that I can find the nodes that need to fire next in sequence.
            for determinedConnection in self.layersSynapse[len(self.layersNeuron) - 1]:# from the previous layer of connections...# If it's not on the list,add it.
                if not any((determinedConnection[0], determinedConnection[1]) == x for x in currentLayerInputIDs):# copy only the unique IDs. No duplicate computations of Nodes.
                    currentLayerInputIDs.append((determinedConnection[0], determinedConnection[1]))# add it to the list of input locations. Ie, next Node layer.
                    if any((determinedConnection[0], determinedConnection[1]) == x for x in lastLayerInputIDs):# if current input ID matches an input ID in the previous layer, promote the node.
                        self.layersNeuron[len(self.layersNeuron) - 1].remove((determinedConnection[0], determinedConnection[1]))# remove the node ID from the previous layer of self.nodeLayerStructure
                if determinedConnection[0]:#if connection has input from an internal neuron(, as opposed to input from input vector).
                    self.layersNeuron[len(self.layersNeuron)].append(determinedConnection[1])# currentLayer.append(internalNodeID)
            # Build connection layer
            workingLayerSynapses = list()
            for connection in backBurner:# for every connection in the back burner# Make the next layer
                # the drop off location can't be an output node. because on the first run everything with an output Node as their output will already be put away.
                if any((connection[2], connection[3]) == id for id in currentLayerInputIDs):# if the connections' output matches an input ID from a connection in the previous layer:
                    workingLayerSynapses.append(connection)# it is part of the next layer.
            if workingLayerSynapses:# if there was something there, add it to the stack.
                self.layersSynapse.append(workingLayerSynapses)
                if not len(backBurner):
                    needToBuildNextLayer = False
            else:# Layer was empty, no need to build a next layer.(it would just come up empty ad infinitum.)
                needToBuildNextLayer = False# Brain finished, stop building. Last layer was the top layer
        self.layersSynapse.reverse()
        self.layersNeuron.reverse()

    def __sigmoid(self, x):# Sig and Tanh take (-4,4) but Sigm gives (0,1) and Tanh gives (-1,1)
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
        ''' It returns zero if the input is negative, otherwise it returns the given input.'''
        result = []
        for i in x:
            if i < 0:
                result.append(0)
            else:
                result.append(i)
        return result

    def proccessInternalNodeLayer(self, workingLayer:'list[int]'):# Layer here is a list of IDs referencing specific Neurons in self.internalNeurons which must be proccessed.
        #generate list of float(-1,1) aka result of tanh
        '''Updates internal Nodes' states in place.\nreturns None.'''
        for i in range(len(workingLayer)):# for every Node in the provided layer of IDs, set the output state of that Node while also resetting it's input cache.
            self.internalNeurons[workingLayer[i]] = perceptron(self.internalNeurons[workingLayer[i]], self.__tanh)

    def proccessFinalNodeLayer(self):
        '''return outputVector\n
        as a 1-D vector of binary returns from the binaryStep activation function.'''
        outputVector = list()
        for i in range(len(self.outputNodes)):# make sure to use the right activation function
            outputVector.append(self.__binaryStep(perceptron(self.outputNodes[i], self.__sigmoid) - 0.92))## have this threshold start low(even so far as 0) and increase infinitely ever closer to 0.95(or maybe even .98, but I wouldnt go higher...) as generation count increases.
        return outputVector

    def think(self, inputVector:'list[float]'):# forward pass # The neural network thinks.
        '''return list[single-bit binary]'''# I need to produce an output vector from the whole network. List of Binary.# Prepare input set for our neural network.
        # Inputs range from 0-1!)
        thinkingLayerNeurons = int()# If I need to run nodes first, do so.
        if len(self.layersNeuron) == len(self.layersSynapse) + 1:# If there's one less synapse layer, Node layer gets priority. Else they're the same value to start with and a synapse layer runs first. Otherwise, something broke...
            self.proccessInternalNodeLayer(self.layersNeuron[thinkingLayerNeurons])# run the first node layer before starting the connection computations
            thinkingLayerNeurons += 1 ## find a way to remove this line and the other one ^.(if possible)
        for connLayerIndex in range(len(self.layersSynapse)):# go back and forth between running a connection layer and a node layer. For each Cycle layer...
            for synapse in self.layersSynapse[connLayerIndex]:# find out what you are mutiplying by this weight: Working Input.
                ## what if chance finds a way to try to store into a nonexistant node?
                if synapse[0]:# if input's coming from an internal neuron
                    result = mult(synapse[4], self.internalNeurons[synapse[1]])# get working input from which one.
                else:# if input's coming from a raw input
                    try:
                        result = mult(synapse[4], inputVector[synapse[1]])# get working input from which one.
                    except:# "List Index Out of Range":
                        result = mult(synapse[4], 0)
                if synapse[2]:# if output is internal# store result in the cache of the propper node.
                    self.internalNeurons[synapse[3]][0].append(result)# referencing: (not always )empty list
                else:# is output neuron
                    self.outputNodes[synapse[3]][0].append(result)# synapse[oddNum] returns an ID.
            # until the last node layer which needs to be skipped
            if not connLayerIndex == len(self.layersSynapse) - 1:# When calculating the last synapse layer, I know the next Node layer is the output one.( I don't need to include output layer in the node structure.(But I do it anyway(someone help me)))
                self.proccessInternalNodeLayer(self.layersNeuron[thinkingLayerNeurons])# After every synapse in the layer has been calculated, The next layer of neurons needs to fire.
                thinkingLayerNeurons += 1# gather input values and proccess the layer of Nodes with each advancement.
        return self.proccessFinalNodeLayer(self)# run final output node layer

    def seed(self):
        return self.genome

# MAIN BLOCK
if __name__ == "__main__":
    # We model a simple nn, with 3 inputs, 1 output and one random gene per 10 attempts.
    #Intialise a single neuron neural network.
    neuralnet = NeuralNetwork(1, 1)
    frame = neuralnet.think([random.random(), 0.5, 1])
    print(frame)
    generation = 1
    genePool = list()
    maxGenePool = 31




    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]])## removed .T, why would I want to transpose this??


    # timeit.timeit()# that's not right... maybe just utilise datetime
    # for i in range(10000):### time it
        # for every step of the simulation that the network instance is alive...
        # The Sim will feed an input vector to the nn, as well as reward so it can learn. If it learns to do something that gets it killed, oh well. Must've fallen in with the wrong crowd...
        # with the self brain setup and the current inputVector, think. return chosen action(s) back to Sim.

        # neural_network.think(training_set_inputs), training_set_outputs)

        ### train network

    # Test the neural network with a new situation.

    print(f"Correct answer: \n{training_set_outputs}")
    print(f"Final answer: \n{neuralnet.think(training_set_inputs)}")###
    print(f"Considering new situation [1, 0, 0] -> ?: {neural_network.think(array([1, 0, 0]))}")
######

# I'm confident we cant fully solve the problem presented in the new Situation with only one neuron

# # # # INSPIRATION CODE
# from numpy import sqrt
#
# # DEFINE THE NETWORK
# # Generate random numbers within a bounded normal distribution
# # def truncated_normal(mean=0, sd=1, low=0, upp=10):
# #     return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
#
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
#
#     def create_weight_matrices(self):
#         """ A method to initialize the weight matrices of the neural network"""
#         rad = 1 / sqrt(self.no_of_in_nodes)
#         X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
#         self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes, self.no_of_in_nodes))
#         rad = 1 / sqrt(self.no_of_hidden_nodes)
#         X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
#         self.weights_hidden_out = X.rvs((self.no_of_out_nodes, self.no_of_hidden_nodes))
#
#     def train(self, input_vector, target_vector):
#         pass # More work is needed to train the network
#
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
#
# # # RUN THE NETWORK AND GET A RESULT
# # Initialize an instance of the class:
# simple_network = Nnetwork(no_of_in_nodes=2,
#                                no_of_out_nodes=2,
#                                no_of_hidden_nodes=4,
#                                learning_rate=0.6)
# # Run simple_network for arrays, lists and tuples with shape (2):
# # and get a result:
# simple_network.run([(3, 4)])



# # # # # DEV NOTES
# the comment key I always go by
# (hopefully )Informative comment.
## something to come back to when the project is complete.
### TO-DO
#### probably an accident.
##### Bookmark. I had to leave off here.
# yep... Thaaat's my system...

# Just thought:
# put the agent in a client connection for a socket based system
# so the environment runs in the server.
# what does the environment do again?
# cuz the agents recieve observations( in the form of a list of floats)
# and return/send outputs( as a list of bool?)

# # so I have two options, right?
# # the one where each layer is calculated before moving to the next
# # and the way that guy did it...
# # where instead of the inputs being passed down the layers and transformed along the way...
# # instead, the inputs are just there and the connections are calculated in order.
# I think I've created a combination of the two concepts
# if so, I've surely done it by structuring the network with all the outputs neurons in the last lasyer, first.

# FROM NEAR THE END OF NeuralNetwork.__init__
            # in one run of this code up to this point should leave us with
            # build output layer from outputCount and genes[bias]
            # the layer X1
            # decode genome to get all Connections
            # init all internal Nodes from output IDs of decoded Connections
            # build base connection layer from output layer and internal nodes
            # Layer Y1
                # if no connections left or no connections output to outputs: brain done
            # loop
            # loop: i += 1
            # cycle through remaining connections collecting those whos output IDs are in list of last layer input IDs.
                # get input IDs from previous layer of Connections
                # build next internal Neuron layer from input IDs and internal Nodes
                # Layer Xi+1
                # build next Connection layer from input IDs and remaining Connections
                # Layer Yi+1
                    # if no connections left to look at or no connections output to prevLayerInputs: brain done
                    # so, it stops when it either:
                    #     runs out of connections
                    #     or ends the loop with an empty layer of connections


# def mult(weightValue:'float', input):
#     '''return float'''
#     ## if BIAS == 'synapse':
#         # you get the idea?# yeah, but I'm NOT doing that type of thing here...
#     return input * weightValue
    

# for netowrks with parents, a genome should be supplied by the Sim( pulled from the avaliable population).
# Each gene should come with a bias for each internalNeuron whether that is the gene that initialised the neuron or not.
# a genome doen't know how many internal Neurons it has until it is built into a brain

