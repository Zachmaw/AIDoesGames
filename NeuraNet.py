# IMPORTS
from numpy import exp, exp2, tanh#, random, array, heaviside, dot, transpose, sin
# from commonFuncs import hextobin
### import timeit

# CONSTANTS INIT
BIASES = list((i-8) * 0.5 for i in range(16))# could only get range(-4, 4, 0.5).
# considerations: We mean the weights AND BIAS before applying the activation function. 
# I would like to apply larger cost to neurons with stonger biases, as they are more likely to be "stubborn" and not learn as well. But I don't want to punish them too much, as they may be needed for some reason. So I will apply a cost of 0.01 per bias point. This is a small cost, but it will add up over time. The weights will have a cost of 0.005 per weight point. This is a smaller cost, but it will add up over time as well.

# from comonFuncs \/
def binToHex(binaryString):
    return hex(int(binaryString, 2))
def hextobin(hexaString):
  return bin(int(hexaString, 16))[2:].zfill(len(hexaString) * 4)

# CLASSES
class NeuralNetwork():
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
    def __RELU(self, x):
        ''' It returns zero if the input is negative, otherwise it returns the given input.'''
        result = []
        for i in x:
            if i < 0:
                result.append(0)
            else:
                result.append(i)
        return result

    def _perceptron(self, node:"tuple[list[float], float, float]", activationFuncOfChoice):# AKA, it shouldn't have an ID by this point. No lookups. Shape : tuple(list[inputs], bias, state).
        '''Returns the node you pass in, but "cleaned" in a way.\n
        Reset to a usable state, while also storing this result.\nConverts weights through Bias into State.'''
        combinedFloats = float()# add up the
        for foat in node[0]:# inputs which are
            combinedFloats += foat# already stored
        combinedFloats += node[1]# and add bias
        thing = len(node[0]) + 1# average the inputs and bias
        # the result of a node is a float no matter what, internal(-1,1), output(0,1). But then we can interpret those however we want.
        return (list(), node[1], activationFuncOfChoice(combinedFloats / thing))

    def _decode_gene(self, hexGene: str):
        '''One hex gene -> one decoded synapse tuple:
        (inputIsInternal, sourceID, outputIsInternal, destID, weight, bias)'''
        bitGene = hextobin(hexGene)
        return (
            int(bitGene[0]),                                  # input type. 0=external, 1=internal
            int(bitGene[1:8], 2),                              # input source ID. 7 bits (0-127)
            int(bitGene[8]),                                   # output type. 0=output, 1=internal
            int(bitGene[9:16], 2),                              # output destination ID. 7 bits (0-127)
            float((int(bitGene[16:32], 2)+1) / 6555)-4,          # weight value. 16 bits (0-65535)
            BIASES[int(bitGene[32:36], 2)]                     # bias, looked up from the preset table. [-4, 3.5]
        )
    def _partition_genes(self, outputCount: int):
        '''Decode every gene once and split them into:
          - enProgressConnLayer: synapses that go straight to an output node
            (this becomes brain[0]'s first/shallowest layer)
          - inputNID_SC: the internal node IDs those synapses reference
            (this becomes brain[1]'s first/shallowest layer)
          - backBurner: synapses that feed ANOTHER internal node instead of
            an output -- these get resolved into deeper layers afterward,
            once we know which connections actually end up needed.
        '''
        backBurner = list()
        inputNID_SC = list()
        enProgressConnLayer = list()
        for hexGene in self.genome:
            synapse = self._decode_gene(hexGene)
            if synapse[2]:# destination is another internal node -- come back to it later
                backBurner.append(synapse)
            else:# destination is an output node -- add it to the current layer
                if synapse[3] +1 <= outputCount:# destination output ID in range
                    enProgressConnLayer.append(synapse[:-1])# drop the bias -- only needed for internal nodes
                    if synapse[0]:# source is an internal node -- make sure it exists and is tracked
                        self._generateNode(synapse[1], synapse[5])# using ID and Bias.## Could return False if an earlier connection instantiated the same internal node, but that's fine -- we just want to make sure it exists.
                        if synapse[1] not in inputNID_SC:
                            inputNID_SC.append(synapse[1])
        return enProgressConnLayer, inputNID_SC, backBurner
    def _build_hidden_layers(self, backBurner, inputNID_SC):
        '''Repeatedly scan backBurner, pulling out any synapse whose
        destination was needed by the layer we just finished, one layer
        deeper each pass, until nothing new is left to add.'''
        while backBurner and inputNID_SC:
            enProgressConnLayer = list()
            lastNodeLayerIDs = inputNID_SC
            inputNID_SC = list()
            for conn in backBurner:
                if conn[3] in lastNodeLayerIDs:# if we need it
                    enProgressConnLayer.append(conn)# add it
                    if conn[0]:# source is internal -- create it, or promote it if it
                        if not self._generateNode(conn[1], conn[5]):# already exists in a shallower layer.
                            self._promote_node(conn[1])
                        if conn[1] not in inputNID_SC:
                            inputNID_SC.append(conn[1])
            if not enProgressConnLayer:
                break# nothing in backBurner connected to the previous layer
            self.brain[0].append(enProgressConnLayer)
            if not inputNID_SC:
                break# last computed layer's nodes are all fed directly by external inputs
            self.brain[1].append(inputNID_SC)
    def _promote_node(self, nodeID) -> bool:
        '''A node that already exists but is needed one layer deeper than
        where it currently sits gets removed from its current (shallower)
        layer here -- the caller re-adds it to the new, deeper layer
        immediately after. Returns whether it was found and removed.'''
        for layer in self.brain[1]:# every layer built so far is shallower
            if nodeID in layer:
                layer.remove(nodeID)
                return True
        return False
 
    def __init__(self, myGenes:'list[str]'=list(), outputCount:'int'=1):
        self.genome = myGenes
        self.outputNodes = [(list(), float(), float()) for _ in range(outputCount)]# Shape: tuple(list[float(results from weights)], float(bias), float(status(-1,1)))
        self.internalNodes = dict()# An internal neuron is tuple and contains workingInputs, bias and state. tuple(list[float(-4,4)]), float(-4,3.5), float(-1,1)). Shape : {ID:(inputs, bias, state)}
        self.brain = (# Fully decoded connections and internal neuron IDs, stacked in layers. # AKA the Connectome.
            list(),# Shape: list[list[tuple(int, int, int, int, float)]] = series of batches of synapses,
            list())# Shape: list[list[int]] = series of batches of internal node IDs, except for the last layer which is the output layer.
        enProgressConnLayer, inputNID_SC, backBurner = self._partition_genes(outputCount)
        if enProgressConnLayer:
            self.brain[0].append(enProgressConnLayer)
        if inputNID_SC:
            self.brain[1].append(inputNID_SC)
        self._build_hidden_layers(backBurner, inputNID_SC)
        self.brain[0].reverse()
        self.brain[1].reverse()
    def _generateNode(self, iD, bias) -> bool:# it needs to know the ID of the Node it is generating, so it can be referenced by other Nodes. It also needs to know the bias.
        '''Returns True if the Node was successfully Generated\nand False if it already exists.'''
        if iD not in self.internalNodes:# if that Node ID already exists, skip building it.
            self.internalNodes[iD] = ([], bias, 0.0)# generate internal node from Conn source ID. Shape: ID = tuple( list[weighted values], bias, state)
            return True
        return False
    def _proccessInternalNodeLayer(self, workingLayer:'list[int]')-> None:
        '''Updates internal Nodes' states in place.'''
        for nodeId in workingLayer:# for every Node in the provided layer of IDs, set the output state of that Node while also resetting it's input cache.
            self.internalNodes[nodeId] = self._perceptron(self.internalNodes[nodeId], self.__tanh)
    def _proccessFinalNodeLayer(self) -> 'list[float]':
        '''Returns the outputVector as a list of floats in range(0,1).'''
        outputVector = []
        for i, node in enumerate(self.outputNodes):
            self.outputNodes[i] = self._perceptron(node, self.__sigmoid)
            outputVector.append(self.outputNodes[i][2])
        return outputVector
    def _calcConn(self, synapse, inVec) -> None:
        '''Please pass in the entire decodedSynapse\nAnd the entire input vector.'''
        if synapse[0]:# if input's source is an internal neuron
            try:
                result = synapse[4] * self.internalNodes[synapse[1]][2]
            except KeyError:
                result = 0
        else:# source is a raw external input
            try:
                result = synapse[4] * inVec[synapse[1]]
            except IndexError:
                result = 0
        if synapse[2]:# if destination is internal. Store result in the cache of the propper node.
            self.internalNodes[synapse[3]][0].append(result)
        else:# destination is an output neuron.
            self.outputNodes[synapse[3]][0].append(result)
    def think(self, inputVector:'list[int]')->"list[float]":# forward pass # The neural network thinks.:
        '''Full forward pass. Returns list[float(0,1)].'''
        offset = len(self.brain[0]) - len(self.brain[1])# if the number of synapse layers is greater than the number of node layers, we need to offset the synapse layers by 1. Normal Use.
        if offset not in (0, 1):
            raise Exception("It very broke.")
        if offset == 1:# extra synapse layer with no paired node layer. Happens when things are normal.
            # Layers are even only if at least one used node doesn't reach the input layer.
            for conn in self.brain[0][0]:
                self._calcConn(conn, inputVector)
        for synapseLayer, nodeLayer in zip(self.brain[0][offset:], self.brain[1]):
            self._proccessInternalNodeLayer(nodeLayer)
            for conn in synapseLayer:# for every connection in the current synapse layer, calculate the result and store it in the proper node.
                self._calcConn(conn, inputVector)
        return self._proccessFinalNodeLayer()
    def seed(self)->"list[str]":
        '''returns a genome ready to have speed appended to the front of each gene'''
        return self.genome
    def getCost(self, cost_scale_conn=0.005, cost_scale_node=0.01)->tuple[float, float]:## could try 0.05 and 0.1
        if not cost_scale_conn and not cost_scale_node:
            return (0, 0)
        connection_cost = 0.0
        for layer in self.brain[0]:
            for _ in layer:# cycle through every connection in every layer of the completed brain
                # temp = abs(conn[4]) * cost_scale_conn## feels iffy
                # connection_cost += temp
                connection_cost += cost_scale_conn
        neuron_cost = 0.0
        for node in self.internalNodes.values():
            temp = abs(node[1]) * cost_scale_node
            neuron_cost += temp
        return (connection_cost, neuron_cost)































































# MAIN BLOCK
if __name__ == "__main__":
    # I'd like to model a simple nn, with 3 inputs, 1 output and one additional random gene per generation.
    # Remember that 'gene' = 'connection'.
    # The first MANY generations won't even *have* connections to inputs...
    # If the initial phase is always the same, why not skip it?
    # What are you proposing? Generate a random number of random genes to begin? What would that prove?
    # Dunno, but I should do it. Just to get it going...
    # Do we have to utilize virgin generation?
    # I could say: if genome length lessThan generationCount + 1, Add a gene.





#     parentGenes = testNet.seed()
#     for i in range(generationCap):# per each generation...
#         currentGenerationPopulation = list()# of NNs
#         for n in range(popsPerGeneration):# per each NN in the generation...
#             # make a NN
#             neuralnet = NeuralNetwork()
#             # Many environments will require many NNs to be alive at once, so...
#             # many environments will require multiple passes of the same NN, so...
#                 # I should keep a list of them.

#             # ask it a question/game
#             # if it's answer/score is in at least nth place( what, top 20%?)
#                 # Save it's genome as a parent
#             # kill it
#             # 



#             # path, shouldEquali = uniquify("GenePools\\testPool\\test_.txt")
#             # with open(path, "w") as f:
#             #     f.write(f"Your GENOME goes here\n{path}\n{i} : {shouldEquali}")
#                 ### next step looks like GA shit...maybe?# The Sim can do it!
#                 # I have both mutations happen in NN init.
#                 # I can almost guarentee cloning, so if I want a clone, I should clone 2 backups.
#                 # Otherwise, I have radiation and Toxic waste to aid in mutation.
#                 # So what you're saying is, alterations to the Genome don't occur until
#                 # a NN is being initialised( with rads and toxins).
#                 # I can't populate a gene pool.
#                 # I have to just keep the parent genome and generate a NN each time I need one?
#                 # That almost seems better, no?
#                 # Because the only genomes that are gonna be kept for parents
#                 # are the best ones from the previous run of the Environment.
#                 # Yeah, that seems like a way better system that generating a gene pool.
#                 # so that means I need to:
#                 # Save the "seed" of the best/victorious Agents

#     # The training set. We have 4 examples, each consisting of 3 input values
#     # and 1 output value.
#     training_set_inputs = array([[1, 1, 1], [1, 0, 1], [0, 1, 1]])
#     training_set_outputs = array([[0, 1, 1, 0]])## removed .T, why would I want to transpose this??


#     # timeit.timeit()# that's not right... maybe just utilise datetime
#     # for i in range(10000):### time it
#         # for every step of the simulation that the network instance is alive...
#         # The Sim will feed an input vector to the nn, as well as reward so it can learn. If it learns to do something that gets it killed, oh well. Must've fallen in with the wrong crowd...
#         # with the self brain setup and the current inputVector, think. return chosen action(s) back to Sim.

#         # neural_network.think(training_set_inputs), training_set_outputs)

#         ### train network

#     # Test the neural network with a new situation.
#     print(f"Correct answer: \n{training_set_outputs}")
#     print(f"Final answer: \n{neuralnet.think(training_set_inputs)}")###
#     print(f"Considering new situation/Environment [1, 0, 0] -> ?: {neuralnet.think(array([1, 0, 0]))}")
# # I'm confident we cant fully solve the problem presented in the new Situation with only one neuron

















# # # # # DEV NOTES

# the comment key I always go by
# # Note to any reader.
# ## Note to Author
# ### TO-DO
# #### probably an typo.
# ##### Bookmark I had to leave off at. Unfinished.



### So, do I include the Output node layer in the brain structure? So, no. We keep the Output Node Layer seperate from the brain( which feels weird, but whatever).
# option A: yes
    # Then when thinking, default layerCount is equal
    # while offset is one less synapse layer.
# option B: no
    # When thinking, there's normally one more synapse layer than Neuron layers



### mutation prefference settings. synapse vs neuron.
### cost.


# We are decoding the genes as weights and storing them in layers based on their I/O targets.
    # you find out how many connections you have to output nodes, placing those in the final layer of the synapse structure first, setting the rest aside for now.










# Just thought: For MUCH later...
# put the agent in a client connection for a socket based system
# so the environment runs in the server?
#





# Each gene also comes with an initiative gene which is one character long right at the beginning.
# this bit, which has a range of 16(0-f(15)), either raises or lowers, per gene, the initiative order of the Agents in a multi-Agent episode.
# Then the number of genes determines a small positive or negative bonus which is then also applied to initiative.
### This gene is trimmed in decoding.

