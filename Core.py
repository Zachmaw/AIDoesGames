


# Obtain settings regarding the size/position of the game window.
# call a selected environment from the environments folder
# The Simulation initializes an Environment, checking it's maxPopulation.
# load a nn from text file or pickle or something
# The Sim loads the population of availiable Genomes.
# The Sim adds all players to the Environment, starting with User Agents followed by NN Agents.
# The Sim selects from the population of Genomes based on the Env's max population.
#

# the Simulation communicates between Agent and Environment.# what do you mean by that?
# because like the Agent and the Environment are seperate...# duh
# So I mean, We run Core.py
# Sim says, "Okay, Human. Which Environment will we be running Agents in, today?"
# Sim loads the specified environment. currentEnv = 




# I wrote on phone:
# we need to:
    # select an environment
        # Environments vary in max Agents per run
    # load/generate a population of NN with feature limits imposed by the selected Env
        # If generating, the number of genes per genome should be balanced with the number of attempts, think log7?
    # Enter a loop:
        # whiteboard
        # Agents all are given the input vector
            # first step in the Env all Agents recieve empty inputs
            # and the outputs are given to the Environment so it can take it's first turn







import pygame, os
from sys import exit
import random
from NeuraNet import NeuralNetwork
from environments.base import *







def pickEnv(name):
    klass = EnvList(name)
    return klass()




waitingPlayers = list()




class Player:
    def __init__(self, id) -> None:
        self.ID = id
        pass

    def getPressed(self):
        return pygame.key.get_pressed()

    def queueUp(self):
        waitingPlayers.append(self.ID)




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
    def __init__(self, outputCount:"int", generation:"int", memX:"int"=5, memY:"int"=5, geneSeq:'list[str]'=None) -> None:
        self.memory = list()
        for y in memY:
            self.memory.append(str())## alternatively .append("")
            for x in memX:
                self.memory[y] += "0"
        if geneSeq:### Build a NN from a Genome to handle the object.
            self.nn = NeuralNetwork(outputCount, generation, geneSeq)
            pass
        else:### No genome given, Check if any humans are waiting to play
            if not len(waitingPlayers):# if no players waiting, proceed
                pass
            pass### Make the object recieve input from input devices( wait on the User)
            # THIS Agent is a Player

    def rollInitiative(self):
        # self.initiative =
        pass




EnvList = {}
# name_to_class = dict(some=SomeClass,
#                      other=OtherClass)

for (dirpath, dirnames, filenames) in os.walk(os.path.join(os.path.realpath(__file__), "environments")):
    for f in filenames:### make sure we don't add Base to that dict
        if not f == "base.py":
            EnvList[str(f)[:-3]] = ### HOW??? Gotta assign the specific class names...
    break
# Does each Env NEED to be a class?
# all Environments inherit from the base Env class because they all need to
# remember their own internal state
# uhh, no. That's *why* they're a class, not why they inherit from one.


# Let's just get one working
# 
# class numberGuess()



currentEnv = EnvList[str(input())]



class Sim:
    def __init__(self, game:'str') -> None:### game needs to be a string
        '''Initialize an environment for this simulation'''
        self.environment = game(players)## set to a user defined class which imports from Env
        self.envHistory = dict()### a place to store kept Environments by a name in str and list of settings?
    def advance(self, actions:'list[int]', envRules:'function'):
        '''Each Agent is always allowed one action per timestep.\n
        It is important to maintain the order of Agent actions.'''
        return envRules(actions)### make sure actions.len() can be varied between timesteps.(as long as len(actions) == len(agents))
    def setEnvironment(self,newEnv,keepEnv:'bool'=False):
        '''Set a new environment for this simulation
        (Almost always incredibly deadly for Agent objects...)'''
        # ### keepEnv
        # if keepEnv:
        #     pass### ugh
        self.environment = newEnv
    
    def addAgent(self, agentID:"tuple(int)", speed:"int"):### why is that a tuple? how big is it suposed to be?
        self.init_order.append(speed, str("LairAction"))
        # sorts by initiative roll
        self.init_order = sorted(self.init_order, key=itemgetter(1), reverse=True)
        #####
        # What I need to do is either put the AgentID in the initiative order list
        # or put the Agent itself in the list?
        # well, when I build the NN from a saved txt file Genome, I have to store it in Sim memory as a list named initOrder.
        
        # Generate NN, store it in dict with key as f"NN{playerNumber}"





        ### IMPLEMENT SPEED GENE
        # NNs can clone, so technichally I don't HAVE to ever keep parents alive, right?
        # If that's the case, I can delete the selected genome from the gene pool the moment I build it into a NN?
        # When I build the NN, place it in initiative. but it needs a speed value...
        # It's time to extend the genome again, I guess... This gives me four bits
        # Let's put it at the beginning and trim it off before it get's to decodeBitstring()
        # Using these 16 permutations, I have 16 values for my initiative order...
        # Env should always have a value of 0 16 or 17, depending on sorting and the binary thing...
        # This speed value should be passed to addAgent along with the same Agents ID.
        #
        # I need to have a gene pool to reference with agentID
        # A gene pool can be:
        # A seperate folder for each Environments gene pool.
        # Where each genome in the pool/folder is a txt file.
        # where each gene in the genome/txt is represented as a string of hexdec characters.
        #
        # The next thing is the naming method for Genomes in storage.
        # The only Genomes in storage are the successful/best ones.
        #####

        # with open("GenePools\\testPool\\")
        #     avaliableGenePool =




# class Simulation:
#     def __init__(self, environment:'Env') -> None:
#         self.environment = environment
#         self.running = False
#     def advance(self):
#         pass
#     def run(self):
#         # repetedly call advance
#         while self.running:
#             self.advance()
#         pass






# MAIN BLOCK
if __name__ == "__main__":
    pass








## gradient descent... I didn't end up using. I went with Genetic.
# from numpy import exp, array, random, dot
# training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
# training_set_outputs = array([[0, 1, 1, 0]]).T
# random.seed(1)
# synaptic_weights = 2 * random.random((3, 1)) - 1
# for iteration in xrange(10000):
#     output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
#     synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
# print( 1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights)))))
