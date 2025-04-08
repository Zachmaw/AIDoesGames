

# Sim coordinates information between
# Env and Agent population
 
# I dunno about to start with, but
# we need to:
    # select an environment
        # Envs vary in maxNN per run
        # some Envs can only take one NN per run.
    # load/generate a population of NN
    # enter a loop where:
        # whiteboard
        # the Agent selects the action(s) to take
        # which is returned
            # on the first step in the Env, Agent recieves empty inputs as no actions have yet been taken? Why can't shit just depend on the env...
        # 
        
        
class Sim:
    def advance(self):
        '''Advance the Simulation by one "step"\n
        that means go down the initOrder\n
        and update gamestate with each Agents input.'''
        pass
        for speed, agentID in self.initOrder:
            newObservations = self.environment.recieveData?()
        





 
# the NN is being stored in the init order list in the Sim class
# so I need a class function of Sim to call 
 
 
class Sim():
    def addPlayer():
        pass
# when the Sim adds an Agent it needs to:
# generate the Agent, with genome based on current generation,
# storing it in playersDict with an ID, incrementing by one.
# then initOrder.append((Agent.speed, agentID))
 
 
 
def decodeSpeed(hexdecSpeedGene:"str"):
    '''Alrighty... How do I want to do this?
    I want to make it so that the number 
    of characters in the speed gene affects
    the range of speeds available to that genome
    then I can just average out the hexdec digits
    by their index and transform the result with a
    coresponding speed list which has been cropped.
    How do I make a graph to indicate what I want...
    I need to know what slopes I want until what points.
    I need to know my x and y axiis.
    x axis = number of genes in the genome
    y axis = 'bonus' to speed value
    we can cap y at 0 and 1
    x is 0 to unbounded
    
    take the average, range(0,15)
    apply balance bonus to all, +3
    apply bonus based on gene count, +-2
    newRange = (1,20)'''
    
    pass
 
 
 
# on an Agents initiative, 
# we pass to the Agent the current observations
# which we get with
class Sim:
    def advance(self):
        actionVector = self.agents[f"NN{self.initOrder[self.currentInitiative][1]}"].nn.think(
        )### this is the part where I gather the observations.
# we recieve from the Agent it's actionVector
# pass actionVector to Sim.environment.proccess?
# should update the environment gamestate in place.
 
 


 
 
 
class Sim:
    def __init__(self, game:'str') -> None:### game needs to be a string?
        # self.playerCount = self.environment.playerCount#####
        ### Nah bro, just check that when I'm generating them.
        # self.envHistory = dict()## a place to store kept Environments by a name in str and list of settings?
        self.agents = list()# container for all Agents in the Sim
        self.initiativeOrder = list()
 
        # self.initiativeOrder.append(50, "LairAction")
        # lair actions are only needed if an Env can
        # modify its own internal state.
 
 
    def addAgent(self, agentID:'tuple(int, int)', environmentString:"str"):### load agent from genome into player dict, giving it a temporary 'system ID'. If it gets selected for reproduction, it will recieve a new ID and be saved.
        genome = loadGenome(agentID, environmentString)
        
 









# when I write a gamerule, i need to know
# how many points of data it should recieve from Agents
# which is just how many things can an agent do?
# this would be where the Agent outputs are defined
# In Pig, this would be two. greed or no greed.
# cant be reduced to one output.
# In ToS, you have abilities that open UIs...
# Agent needs to be able to... at most(necromancer)
# select a player number, that's what, 14 right there
# you need to select a second target
# and then whether to use the extra ability...
# so each role needs a different setup.
# i can say at most and go from there
# necro needs a dead target and a live one
# as well as whether or not to ghoul if book.
# 14 buttons for players in graveyard.
# 14 buttons for players alive and well.
# one for ghoul.
#
# sometimes a role allows the player to target themself,
# but those roles dont allow you to target multiple.
class ToS2(Env):
    def __init__(self):
        self.playerCount = 15
        self.roles = {
            'Vigilante': {'Abilities': 1, }# that doesnt tell if its player selector or bool like alert...

 
        }
# an example of the function which is required to
# accept a certain number of actions:
    # which in this case would be greed or no greed.
# and return observations the current Agent has access to in the given Env:
        # what do you mean by that?
            # sometimes an Agent can only see from its point of view
    # which in this case would be:
        # playerTurnProggress # whos turn it is
        # roundProggress # how many rounds are left
        # thisAgentPlayerNumber # who I am
        # currentTempScore # score on the table # the pot
        # list all player total scores in order
            # amount of Nodes for that = playerCount
class Pig():
    pass
 
# so, Sim can track:
    # initOrder, list of two ints representing init roll and ID/index in players list
    # currentTurn, NOT the index in initOrder
 
 
# full auto...
# how many genomes to attempt per generation
# how many genomes to keep per generation
# how many generations
# 
 
# manual
 
 
 
 
# some games, like ToS, need more to be tracked
# like day/night
# like role abilities
 
 
# name the genome being saved
# as NN[generation]-[savedIndex]
 
 
 
def askPlayer(observations, obvsTypes):
    
    for i in range(observations):
        print(f'{obvsTypes[i]}:{obvservations[i]}')
    try:
        response = input()
        ### check if response is a valid bitstring
        Exception('String is not a valid Bitstring.')
        ### return bitstring as list of bools
    except 'String is not a valid Bitstring.':
        pass
# the something needs to output the Env observations
# as a print statement, so then we have to take a
# string of 1s and 0s through input
# to feed that back to Sim as list of bool
# in place of a NN
# Agent.brain = 'NN' or 'HP'
# def think(self, inputs):
    # if self.brain == 'NN':
        # return self.nn.think(inputs)
    # else:
        # 
        
        
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
######
# self.brain = NeuralNetwork(expectedOutputs:'int', genes:'list[str]', irradiation:"float"=0.01):
# agent.brain.seed()
### Gets called by the Sim when the Network has been selected to aid in repopulation.

 
 
 
# What if the first output node( in sequence) firing means the network wants to update it's memory
# The second and third indicate location(, but by what formatting?).
# And the fourth output indicating what to set to.
# As for location formatting... From two independent floats to a point on a 2D grid.
######
class Agent:# Recieves rewards and observations, and returns an action
    def __init__(self, outputCount:"int"=1, generation:"int"=1, memX:"int"=5, memY:"int"=5, geneSeq:'list[str]'=None, rads:"float"=0.01):
        self.memory = list()
        for y in memY:
            self.memory.append(str())## alternatively .append("")
            for x in memX:
                self.memory[y] += "0"
        if geneSeq:### Build a NN from a Genome to handle the object.
            self.nn = NeuralNetwork(outputCount, generation, geneSeq, rads)
            # shpeeeeeed
        else:### No genome given, Check if any humans are waiting to play
            if not len(waitingPlayers):# if no players waiting, proceed
                pass
            pass### Make the object recieve input from input devices( wait on the User)
            # THIS Agent is a Player
 
    # def rollInitiative(self):
    #     # self.initiative =
    #     pass
 
 
 
# Does each Env NEED to be a class? # I think so, yeah...
# all Environments inherit from the base Env class because they all need to
# remember their own internal state
# uhh, no. That's *why* they're a class, not why they inherit from one.
 
 
 
# which means when this is called by the Sim
# (which is where the Agents are stored)
# currentGeneration, currentEnvironment, and AgentID
def saveGenome(genes, genomeID:"tuple(int, int)", envStr:"str"):
    with open(os.path.join(os.path.realpath(__file__), f"networks\\{envStr}\\{genomeID[0]}\\NN{genomeID[1]}.txt", "w")) as f:
        f.writelines(genes)# Sim.initOrder[currentInitiative].seed()
def loadGenome(genomeID:"tuple(int, int)", envStr:"str"):
    '''returns a list of hexdecimal strings from txt file.'''
    genome = list()
    with open(os.path.join(os.path.realpath(__file__), f"networks\\{envStr}\\NN{genomeID}.txt"), "r") as f:
        for line in f.readlines():
            genome.append(line.strip())
    return genome
 
 
 
        speedSeq = list()
        genome = list()
        for i in range(len(genes)):# run through all the genes in the genome# Decode all the connections into tuples.# for synapse(index) in the_genome:
            temp = mutateHexdec(genes[i], irradiation)# MUTATION * 2 COMBO | By way of irradiation and for growing NNs.
            # the speed gene is never binary, and never recieves binary mutation chance... OH WELL!
            speedSeq.append(temp[0])
            temp = temp[1:]
            bitstring = mutateBitstring(hextobin(temp))
            genome.append(binToHex(bitstring))## (Note to self: Calm down, that's why this is on it's own line..)
            # disect gene
 
        decodeSpeedGene()
 
 
 
 
 
 
 
 
 
 
class EnvBase():
    pass
    
    
    
 
 




























 
 
 
# NOTES
 
 