

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
# Sim loads the specified environment. currentEnv = class


# we need to:
    # select an environment
        # Environments vary in max Agents per run
    # load/generate a population of NN with feature limits imposed by the selected Env
        # If generating, the number of genes per genome should be balanced with the number of attempts, think y=0.75x .
    # Enter a loop:
        # whiteboard
        # Agents all are given the input vector
            # first step in the Env all Agents recieve empty inputs
            # and the outputs are given to the Environment so it can take it's first turn







import pygame, os
# from sys import exit
# import random
from NeuraNet import NeuralNetwork
from environments.base import *
from operator import itemgetter
from numpy import random


from environments.Dice import Pig



def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 0
    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1
    return (path, counter)




def saveGenome(genes, genomeID:"tuple(int, int)", envStr:"str"):
    with open(os.path.join(os.path.realpath(__file__), f"networks\\{envStr}\\NN{genomeID}.txt", "w")) as f:
        f.writelines(genes)# Sim.initOrder[currentInitiative].seed()
def loadGenome(genomeID:"tuple(int, int)", envStr:"str"):
    '''returns a list of hexdecimal strings\n
    but the first one is made of the first character from all of 'em cut off and stuck together\n
    (in order, of course).'''
    speed = list()
    genome = list()
    with open(os.path.join(os.path.realpath(__file__), f"networks\\{envStr}\\NN{genomeID}.txt"), "r") as f:
        for line in f.readlines():
            line2 = line.strip()
            speed.append(line2[0])
            genome.append(line2[1:])
    genome.insert(0, speed)
    return genome

def decodeSpeed(speedGene:"str"):##### what happens if the genome is empty? The speed gene is empty. This needs to be resolved because speed gets decoded before the NN is init'd.

    pass



def geneCountMath(x:"int"):
    '''return a float in the range(0, 0.89)'''
    if x <= 15:# linear(0,0)(15,0.8)
        return 0.055 * x
    elif x > 123:
        return 0.1
    else:
        return -0.0065 * x + 1
 
def fetchBonus(funcOut:"float"):
    '''input range (0, 0.999)'''
    t = funcOut * 5
    if t < 1:
        return -2
    elif t < 2:
        return -1
    elif t < 3:
        return 0
    elif t < 4:
        return 1
    else:
        return 2
 




 
### new mutation algorythm
# all mutating should occur within NN

 
 






### a lot of this should be writen outside the environment... Only pass in decoded actions from the users?
                                                    # yeah, make the Sim dumb it down for NN Agents. Better than making it smart up for users...
        ### define NN then come back# lol this was before NN was defined?
        # self.currentMove = inputs[self.turn]# find out who's turn it is and extract active player action
        # for player in inputs:
        #     result = self.decodeActs(player)
        #     if result == 1:
        #         tempN = self.roll(self.DICE[floor(self.round * 0.5)])#every even round use the next dice up
        #         if tempN == 0:
        #             self.turn += 1### Im not sure that's all I had to do here...
        #         else:# didn't roll 0
        #             self.tempScore += tempN
        #     else:
        #         self.players[self.turn] += self.tempScore
        #         self.turn += 1



EnvList = {##EnvList[f[:-3]]
    'pig': Pig,
    'dice': Pig
}
# name_to_class = dict(some=SomeClass,
#                      other=OtherClass)

def pickEnv(name):
    klass = EnvList(name)
    return klass()##### So, we're returning a called instance of whatever Environment we've chosen,
                        # but always without any arguments.

validEnvNames = EnvList.keys()
# for (dirpath, dirnames, filenames) in os.walk(os.path.join(os.path.realpath(__file__), "environments")):
#     for f in filenames:
#         if f[-3:] == ".py":# Check the extention
#             if not f == "Base.py":# make sure we don't add Base to that dict
#                 validEnvNames.append(f[:-3])### hard code in all the avaliable Environments
#                 # this chunk can add the names to a list so users can only select a valid target from that list.
#     break       # or dict, whatever.

# env = pickEnv()
# currentEnv = EnvList[str(input())]

EnvList = {}
# name_to_class = dict(some=SomeClass,
#                      other=OtherClass)

for (dirpath, dirnames, filenames) in os.walk(os.path.join(os.path.realpath(__file__), "environments")):
    for f in filenames:### make sure we don't add Base to that dict
        if not f == "base.py":
            pass# this was for something else entirely...# EnvList[str(f)[:-3]] = ### HOW??? Gotta assign the specific class names...
    break


# all Environments inherit from the base Env class because they all need to
# remember their own internal state



currentEnv = EnvList[str(input())]




# Does each Env NEED to be a class?
# all Environments inherit from the base Env class because... something.
# they remember their own internal state
# but they need rules by which to update those states.
    #


        # gamestate will be filled with things like:
        # day or night, yes
        # player turn number, Sim can pass it into whatever's being passed to the Agent
        # round/turn/day number, yes, if it's a round based game, it can be handled by the game.
        # various point values, yes like...
        # table score/the pot, ez
        # all players total scores, list with length(all players)



# Let's just get one working
#
# class numberGuess()













waitingPlayers = list()# le HUH?




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
# saveGenome(agent.brain.seed())
### Gets called by the Sim when the Network has been selected to aid in repopulation.

HEX_OPTIONS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F"]

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
def hextobin(hexaString):
  return bin(int(hexaString, 16))[2:].zfill(len(hexaString) * 4)
def mutateHexdec(gene:"str", radiationBonus:"float"):
    '''raises/lowers the value of random bonds by one'''
    for i in range(len(gene)):
        if roll(1000, 994, 1000 * radiationBonus):
            gene = replace_str_index(gene, i, HEX_OPTIONS[int(hextobin(gene[i]), 2) + random.choice([1, 15]) % 16])
    return gene

def decodeSpeed(speedGene:"str"):
    '''Takes the hexStr in from speed gene.
    Decode it to a list of numbers.
    Get bonus from list length.
    Average the list.
    Add the bonus.
    return the total initiative value.'''
    temp = list()
    geneCount = int()
    for hexdecChar in speedGene:
        temp.append(int(hexdecChar, 16))
        geneCount += 1
    bonusInput = fetchBonus(geneCountMath(geneCount))
    return round(sum(temp) / geneCount) + bonusInput
 
# What if the first output node( in sequence) firing means the network wants to update it's memory
# The second and third indicate location(, but by what formatting?).
# And the fourth output indicating what to set to.
# As for location formatting... From two independent floats to a point on a 2D grid.
######
class Agent:# Recieves rewards and observations, and returns an action
    def __init__(self, outputCount:"int"=1, generation:"int"=1, memory:"tuple(int, int)"=None, geneSeq:'list[str]'=None, radiation:"float"=0.01, toxicWaste:"float"=0):
        self.memories = list()
        for x in memory[0]:
            self.memories.append(str())## alternatively .append("")
            for y in memory[1]:
                self.memories[x] += "0"
        if geneSeq:# Build a NN from a Genome to handle the object.
            cleanGenes = list()
            speedGene = list()
            for i in range(len(geneSeq)):# extract speed
                temp = mutateHexdec(geneSeq[i], radiation)
                speedGene.append(temp[0])
                cleanGenes.append(temp[1:])
            del temp
            self.nn = NeuralNetwork(outputCount, generation, cleanGenes, toxicWaste)
            self.speed = decodeSpeed("".join(speedGene))

        else:### No genome given, Check if any humans are waiting to play
            if not len(waitingPlayers):# if no players waiting, proceed
                ### say there were no users queued up, nor was a genome supplied,
                # Therefore an empty NN is being generated.
                pass
            pass### Make the object recieve input from input devices( wait on the User)
            # *This* Agent is a Player instead of a NN

    # def rollInitiative(self):
    #     # self.initiative =
    #     pass





class Sim:
    def __init__(self, game:'str') -> None:### game needs to be a string?
        '''Initialize an environment for this simulation'''
        self.environment = pickEnv(game)### set to a user defined class which imports from Env
        self.playerCount = self.environment.getPlayerCount()#####
        # self.envHistory = dict()## a place to store kept Environments by a name in str and list of settings?
        self.agents = list()# container for all Agents in the Sim # FORMAT: speed:int =
        self.playersCount = int()
        self.initiativeOrder = list()

        # self.initiativeOrder.append(50, "LairAction")


    def addAgent(self, agentID:"tuple(int, int)", environmentString:"str"):### load agent from genome into player dict, giving it a temporary 'system ID'. If it gets selected for reproduction, it will recieve a new ID and be saved.
        genome = loadGenome(agentID, environmentString)
        self.agents[f"NN{self.agents.__len__()+1}"] = Agent(self.environment.actionOptions)#####sleeepy
        ### MUTATE
        self.agents.append(Agent(
            len(self.environment.actionOptions)-1),# how many output Nodes
            
        )
        ### an Agent needs what again?
        # Agent generates the NN so...
        # how many output Nodes
        # the genome - speed genes
        # or wait, the NN mutates itself? shouldnt.
        # but I need to load the genome
        # mutate it
        # take the speed off
        # mutate it again
        # pass that to Agent
        # which passes it to NN
        


# def speeeed(agentObj):
#     return agentObj.initiative

# programming_languages.sort(key=speeeed)


        self.initiativeOrder.append(speed, )### f"NN{agentID[0]}-{agentID[1]}"
        # sorts by initiative roll
        self.initiativeOrder = sorted(self.initiativeOrder, key=itemgetter(1), reverse=True)
        #####
        # What I need to do is either put the AgentID in the initiative order list
        # or put the Agent itself in the list?
        # well, when I build the NN from a saved txt file Genome, I have to store it in memory.
        # Generate NN, store it in dict with key as f"NN{playerNumber}"
        # So list it is...initOrder = list[tuple(speed, agentID)]
        # players = dict{agentID:Agent}

    # def addAgent(self, agentID:"int", speed:"int"):### why is that a tuple? how big is it suposed to be?
    #     self.init_order.append(speed, agentID))
    #     # sorts by initiative roll
    #     self.init_order = sorted(self.init_order, key=itemgetter(1), reverse=True)
    #     #####
    #     # What I need to do is either put the AgentID in the initiative order list
    #     # or put the Agent itself in the list?
    #     # well, when I build the NN from a saved txt file Genome, I have to store it in Sim memory as a list named initOrder.

    #     # Generate NN, store it in dict with key as f"NN{playerNumber}"





        ### IMPLEMENT SPEED GENE
        # Using these 16 permutations, I have 16 values for my initiative order...
        # Env should always have a value of 0 16 or 17, depending on sorting and the binary thing...
        # This speed value should be passed to addAgent along with the same Agents ID.
        #
        #
        # I need to have a gene pool to reference with agentID
        # A gene pool can be:
        # A seperate folder for each Environments gene pool.
        # Where each genome in the pool/folder is a txt file.
        # where each gene in the genome/txt is represented as a string of hexdec characters.
        #
        # For each genome in the Sim, decide if its reward is high enough.
        # After that is the naming method for Genomes to be stored.
        # The only Genomes in storage are the successful/best ones.
        ### but how do they get selected to be put there?
        #
        #
        #
        # well. lets think about it.
        # imagine an Env that takes 2 players taking turns
        # perhaps all Envs are required to declare a func for endStates.
        # chess might have endstate function as
        # player whose turn it is has no legal moves
        
        # imagine an environment that takes one nn.
        # in order to decide whether each given agent should be saved,
        # we should track the performance, save the NN,
        # then compare future NNs to the tracked score
        # if the new score is lower, but maxNNPerGeneration hasnt been reached,
        # save it anyway... 
        # only once max has been reached does overwriting begin
        # remember that the IDs change when saved/loaded.
        # overwrite lowest score only if new score beats it.
        # should use the ID of the overwritten NN
        # which means for each current generation
        # I have to track a list of
        # the saved NNs reward scores with ID as index
        # so that I can overwrite the lowest reward NN by its index in that list
        # when doing so( if current_nn_score > lowestValue(latestGenerationScores:"list"))
        # just save genome as you would normally
        # but pass in the ID of the lowest score
        #
        #
        # how do we know when the environment has reached an endState?
        # examples: when a round limit is reached
        # when the remaining players belong to the same faction? idk... neutrals... have their own 'faction's. its fine
        # what about an environment where agents can join and be removed mid-game? round limit or nothing alive?
        
        #####

    def advance(self, actions:'list[int]', envRules:'function'):
        '''Each Agent is always allowed one action per timestep.\n
        It is important to maintain the order of Agent actions.'''
        return envRules(actions)### make sure actions.len() can be varied between timesteps.(as long as len(actions) == len(agents))
        ### are you suggesting that actions passed in here is
        # a list of ALL Agent actions this initiative round??
        # one action per initiative step, eh?
 




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




    # def setEnvironment(self,newEnv,keepEnv:'bool'=False):
    #     '''Set a new environment for this simulation
    #     (Almost always incredibly deadly for Agent objects...)'''
    #     # ### keepEnv
    #     # if keepEnv:
    #     #     pass### ugh
    #     self.environment = newEnv


### Environments...
# An environment needs to be entirely self contained.
# The Sim will have asked an Agent what it wants to do.
# The Agent responds with a list of bools.
# I guess Sim only passes to Env the first True action in the list.
# update Env internal state with selected action from specified Agent
# Sim expects back an updated set of observations.
# Env needs to be a class, right?
# Just hoping Env doesnt need to be a socketed connection...
 














# ______________________________________________________________________________________

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
