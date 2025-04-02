# IMPORTS
from numpy import random, exp, exp2, tanh, sin, array#, heaviside, dot, transpose
from math import ceil
### import timeit

# CONSTANTS INIT
HEX_OPTIONS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F"]
BIASES = list()
for i in range(16):# could only get -8 to +7, but that's fine because a bias of +8 means that neuron fires(with max power) no matter what.
    BIASES.append(i - 8)

# SETTINGS INIT# future features to be marked with ###
# FUNCTIONS
def binToHex(binaryString):
    return hex(int(binaryString, 2))
def hextobin(hexaString):
  return bin(int(hexaString, 16))[2:].zfill(len(hexaString) * 4)
def diceRoll(d, dc, bonus):
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
def generateMutationBitstring(geneLen:"int", toxicWasteBonus:"float"):
    '''Default odds: 8/1000\nA float of 1.0 should cause mutation chance to be 999/1000,\n
    But only because a Nat 1 is still a possibility.'''
    temp = list()
    for i in range(geneLen):
        if diceRoll(1000, 993, 1000 * toxicWasteBonus):
            temp.append("1")
        else:
            temp.append("0")
    return "".join(temp)
def bitCombine(argA:"str", argB:"str"):# overlay mutation bitstring with gene bitstring
    temp = list()
    for i in range(len(argA)):
        temp.append(str((int(argA[i]) + int(argB[i])) % 2))
    return "".join(temp)
def mutateBitstring(bitstring:"str", bonus):
    return bitCombine(bitstring, generateMutationBitstring(36, bonus))## 36??? Yes, 36.
def perceptron(node:"tuple[list[float], int, float]", activationFuncOfChoice):# AKA, it shouldn't have an ID by this point. No lookups. Shape : dict{ID:tuple(list[bias,input,...], state)}.
    '''Returns the node you pass in, but "cleaned" in a way. Reset to a usable state, while also holding this result.'''
    tempN = float()# add up the
    for floa in node[0]:# inputs which are
        tempN += floa# already stored
    tempN += node[1]# add bias
    result = activationFuncOfChoice(tempN + node[1])# the result of a node is a float no matter what, internal(-1,1), output(0,1). But then we will binary the output nodes.
    return (list(), node[1], result)
def funcSine(generation):# f(x)=mx+A\sin(Bx) m=0.7, A=8, B=0.35 ### setting
    m = 0.7
    A = 8
    B = 0.35
    return m*generation+A*sin(B*generation)

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

    def __init__(self, outputCount:'int'=1, parentGenome:'list[str]'=None, genomeToxicWasteBathPercentage:"float"=0.01, generation:"int"=1):
        '''
        Generation param is only needed if genes are not provided.\n
        (Generation dictates the number of random genes to be generated.)
        '''
        if parentGenome == None:# make some genes based on generation
            parentGenome = init_random_Genome(ceil(funcSine(generation)))
        self.outputNodes = list()# of Output Nodes with Shape: tuple(list[float(results from weights)], int(bias), float(0,1))
        self.internalNodes = dict()# An internal neuron is tuple and contains workingInputs, bias and state. (list[float(-4,4)]), int(-4,3), float(-1,1)). Shape : {ID:(inputs, bias, state)}
        self.brain = (list(), list())# Full decoded Connections, stacked in layers.# Shape: tuple(list[list[tuple(int, int, int, int, float)]], list[list[int]]]
        for i in range(outputCount):# add all output nodes to the first layer of the Neuron structure
            self.outputNodes.append((list(), int(), float()))# Shape: tuple(tuple(list[float(data storage)], int(bias), float(output storage))
        backBurner = list()# unused Synapses
        inNodeIDsSC = list()# Node IDs
        workingLayer = list()# Synapses
        self.genome = list()# for every gene, make it a bitstring, mutate it, then turn it back into hexString and store it in genome.
        for i in range(len(parentGenome)):# Decode all the connections into tuples.# for synapse(index) in parent_genome:
            bitstring = mutateBitstring(hextobin(parentGenome[i]), genomeToxicWasteBathPercentage)# mutation
            self.genome.append(binToHex(bitstring))## Save it back to genome. (Note to self: Calm down, that's why this is on it's own line..)
            decodedSynapse = (# disect current gene
                int(bitstring[0]),# input type. 0 is inputInput, 1 is internal input. 1 bit(0,1)
                int(bitstring[1:8], 2),# input source ID. 7 bits(0,127)
                int(bitstring[8]),# output type. 0 is output, 1 is internal. 1 bit(0,1)
                int(bitstring[9:16], 2),# output destination ID. 7 bits(0,127)
                int((int(bitstring[16:32], 2)+1) / 6555)-4,# weight value. 16 bits(0,65535)
                BIASES[int(bitstring[32:], 2)]# Bias as int representing an index in the list of preset biases. # 4 bits to work with gives me a usable range of 0 to 15, which I map to -8 to +7.
                )
            if decodedSynapse[2]:# if the output type( from this gene) is internal, wait to see if we even need this synapse.
                backBurner.append(decodedSynapse)
            else:# Connects to an output node. Add this Conn to the base Conn layer.
                if decodedSynapse[3] +1 <= outputCount:# if destination Node ID is a valid Output ID.
                    workingLayer.append(decodedSynapse[:-1])# it goes in the last layer.
                    if decodedSynapse[0]:# if the referenced Node is internal, create it so it can be found later.
                        self.generateNode(decodedSynapse)
                        if not any(decodedSynapse[1] == nodeId for nodeId in inNodeIDsSC):# if the referenced node isn't already, add it to the shopping cart.
                            inNodeIDsSC.append(decodedSynapse[1])# add all internal Node IDs for the next layer.
        self.brain[0].append(workingLayer)# take the base Conn layer[every Connection that goes to an output Node] and add it to allSynapseLayers first.
        self.brain[1].append(inNodeIDsSC)
        needToBuildNextLayer = True
        if not len(backBurner):# ran out of Connections to make layers with. Head empty.
            needToBuildNextLayer = False
        elif not len(inNodeIDsSC):# no internal Nodes were connected to.
            needToBuildNextLayer = False
        while needToBuildNextLayer:# try to build a new set of layers[1:]
            workingLayer = list()
            lastNodeLayerIDs = inNodeIDsSC# only gets internal Node IDs that are used in the previous layer
            inNodeIDsSC = list()
            for conn in backBurner:# for every connection in the back burner
                if any(conn[3] == nId for nId in lastNodeLayerIDs):# if the connections' destination ID matches a Node ID from the previous layer:
                    workingLayer.append(conn)# it is part of the next layer.
                    if conn[0]:# if the referenced Node is internal, create it.
                        if not self.generateNode(conn):# if it already exists, promote it.
                            for layer in range(len(self.brain[1])):
                                for nodeID in range(layer):
                                    if self.brain[1][layer][nodeID] == conn[1]:
                                        self.brain[1][layer].remove(conn[1])
                        if not any(conn[1] == nodeId for nodeId in inNodeIDsSC):# if the referenced node isn't already,
                            inNodeIDsSC.append(conn[1])# add all internal Node IDs for the next layer to the shopping cart.
            if workingLayer:
                self.brain[0].append(workingLayer)
            else:# no Synapses joined to the previous Neuron layer...
                needToBuildNextLayer = False
            if inNodeIDsSC:# if there was something there, add it to the stack.
                self.brain[1].append(inNodeIDsSC)
            else:# Layer was empty, no need to try more. All Conns in the working layer attatched to the input vector.
                needToBuildNextLayer = False# Brain finished, stop building. Last layer was the top layer
            if not backBurner:
                needToBuildNextLayer = False
        self.brain[0].reverse()
        self.brain[1].reverse()
    def generateNode(self, para):
        '''Returns True if the Node was successfully Generated\nand False if it already exists.'''
        if not any(para[1] == nodeId for nodeId in self.internalNodes.keys()):# if that Node ID already exists, skip building it.
            self.internalNodes[para[1]] = (list(), para[5], float())# generate internal node. Shape: ID = tuple( list[weighted values], bias, state)
            return True
        else:
            return False    
    def proccessInternalNodeLayer(self, workingLayer:'list[int]'):# workingLayer is a list of IDs referencing specific Neurons in self.allNeurons which must be proccessed.
        #generate list of float(-1,1) aka result of tanh
        '''Updates internal Nodes' states in place.\n(effectively )returns None.'''
        for id in range(len(workingLayer)):# for every Node in the provided layer of IDs, set the output state of that Node while also resetting it's input cache.
            self.internalNodes[workingLayer[id]] = perceptron(self.internalNodes[workingLayer[id]], self.__tanh)### setting: alternate activation function based on layer or node?
    def proccessFinalNodeLayer(self):
        '''Returns the outputVector\nas a list of floats in range(0,1).'''
        outputVector = list()
        for i in range(len(self.outputNodes)):
            outputVector.append(perceptron(self.outputNodes[i], self.__sigmoid))# make sure to use the right activation function
        return outputVector
    def connCalc(self, synapse, inVec):
        '''Please pass in the entire decodedSynapse\nbut only inputVector[synapse[1].'''
        if synapse[0]:# if input's coming from an internal neuron
            result = synapse[4] * self.internalNeurons[synapse[1]]# get data from internal node Node by it's ID in Conn.'
        else:# if input's coming from a raw input
            try:
                result = synapse[4] * inVec# get data from which input.
            except:# "List Index Out of Range":
                result = synapse[4] * 0
        if synapse[2]:# if destination is internal# store result in the cache of the propper node.
            self.internalNeurons[synapse[3]][0].append(result)# referencing: usually not empty list
        else:# is output neuron
            self.outputNodes[synapse[3]][0].append(result)# synapse[oddNum] returns an ID.
    def calcConnLayer(self, layer, inputVec):
        for conn in layer:
            self.connCalc(conn, inputVec[conn[1]])
    def think(self, inputVector:'list[int]'):# forward pass # The neural network thinks.
        '''return list[float(0,1)]\nInputs list[int(0,1)]'''# I need to produce an output vector from the whole network.
        thing = int()
        if len(self.brain[0]) == len(self.brain[1])+1:# if extra synLayer
            self.calcConnLayer(self.brain[0][0], inputVector)
            thing += 1
        if not len(self.brain[0]) == len(self.brain[1]):# if they're anything but even
            raise Exception("It very broke.")
        else:# but they're exactly even.
            for layerIndex in range(len(self.brain[1])):
                self.proccessInternalNodeLayer(self.brain[1][layerIndex])
                self.calcConnLayer(self.brain[0][layerIndex + thing])
        return self.proccessFinalNodeLayer()# run final output node layer
    def seed(self):### does a txt file being read in show line breaks? Requires testing...
        '''returns a genome ready to have speed appended to the front of each gene'''
        return self.genome

# MAIN BLOCK
if __name__ == "__main__":
    # We model a simple nn, with 3 inputs, 1 output and one random gene per 10 attempts.
    # Remember that 'gene' = 'connection'.
    # The first MANY generations won't even *have* connections to inputs...
    # If the initial phase is always the same, why not skip it?
    # What are you proposing? Generate a random number of random genes to begin? What would that prove?
    # Dunno, but I should do it. Just to get it going...
    while True:
        try:
            generationCap = int(input("Generation number to stop at..."))
            popsPerGeneration = int(input("Number of NNetworks per generation..."))
            break
        except:
            print("We got some Non-numbers in there, Friend.\nPlease try again.")
    testNet = NeuralNetwork()
    thought = testNet.think([0.0, 0.5, 1.0])
    print(thought)# Should be in the form of list[bool] with len = NN.outputCount
    parentGenes = testNet.seed()
    for i in range(generationCap):
        for n in range(popsPerGeneration):
            pass
            # make a NN
            neuralnet = NeuralNetwork(1, )
            # ask it a question/game
            # if it's answer/score is in at least nth place( what, top 20%?)
                # Save it's genome as a parent
            # kill it
            # 



            # path, shouldEquali = uniquify("GenePools\\testPool\\test_.txt")
            # with open(path, "w") as f:
            #     f.write(f"Your GENOME goes here\n{path}\n{i} : {shouldEquali}")
                ### next step looks like GA shit...maybe?# The Sim can do it!
                # I have both mutations happen in NN init.
                # I can almost guarentee cloning, so if I want a clone, I should clone 2 backups.
                # Otherwise, I have radiation and Toxic waste to aid in mutation.
                # So what you're saying is, alterations to the Genome don't occur until
                # a NN is being initialised( with rads and toxins).
                # I can't populate a gene pool.
                # I have to just keep the parent genome and generate a NN each time I need one?
                # That almost seems better, no?
                # Because the only genomes that are gonna be kept for parents
                # are the best ones from the previous run of the Environment.
                # Yeah, that seems like a way better system that generating a gene pool.
                # so that means I need to:
                # Save the "seed" of the best/victorious Agents

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
    print(f"Considering new situation/Environment [1, 0, 0] -> ?: {neuralnet.think(array([1, 0, 0]))}")
# I'm confident we cant fully solve the problem presented in the new Situation with only one neuron

















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




# We are decoding the genes as weights and storing them in layers based on their I/O targets.
            # you find out how many connections you have to output nodes, placing those in the final layer of the synapse structure first, setting the rest aside for now.


# note walkthrough...

# class NN
    # init function
        # if no parent
            # make genome from generation count
        # after that I init 5 self.containers
        # and 5 more plain containers...
        ### limit completed bits to self.brain
        # Shape: Tuple(List[List[Synapse]], List[List[nodeID]])
        #

    # think func
        # check layer counts for exactly one more synapse layer
            # pass
        # check for balanced layers
            # process first node layer
        # else: shit broke somehow...
        # while thinking:# while len(self.brain[])
            # 












# Just thought: For MUCH later...
# put the agent in a client connection for a socket based system
# so the environment runs in the server.
#
# what does the environment do again?
# cuz the agents recieve observations( in the form of a list of floats)
# and return/send outputs( as a list of bool?)
# Every "timeStep" the Sim goes down the initiative order
# at each Agent on the list, asking it's response to currentEnvOuts
# currentEnvOuts is being kept up to date between Agents, isn't it...


# since each genome in the pool is just the successors of the previous generation, as opposed to untested whelplings,
# When building a NN I can make the input genome to that be based on as many parent genomes as I want...
# (again with or without mutation based on waste/radiation)

# Each gene should come with a bias for each internalNeuron whether that is the gene that initialised the neuron or not.
# a genome doen't know how many internal Neurons it has until it is built into a brain

# Each gene also comes with an initiative gene which is one character long right at the beginning.
# this bit, which has a range of 16, either raises or lowers, per gene, the overall speed of the Agent.
# Then the number of genes determines a small positive or negative bonus which is then also applied.