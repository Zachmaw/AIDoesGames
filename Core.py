

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
from NeuraNet import NeuralNetwork
from environments.base import *
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
    but the first one is a list made of the cut off first character from all of 'em\n
    (in order, of course).'''
    genome = list()
    with open(os.path.join(os.path.realpath(__file__), '..', f"networks\\{envStr}\\NN{genomeID[0]}-{genomeID[1]}.txt"), "r") as f:
        for line in f.readlines():
            genome.append(line.strip())
    return genome

def geneCountMath(x:"int"):
    '''return a float in the range(0, 0.89)'''
    if x <= 15:# linear(0,0)(15,0.8)
        return 0.055 * x
    elif x > 123:
        return 0.1
    else:
        return -0.0065 * x + 1
def fetchBonus(funcOut:"float"):
    '''Input range(0, 0.89)
    Output range(-2, 2)'''
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
def decodeSpeed(speedGene:"str"):
    '''Takes the hexStr in from speed gene.
    Decode it to a list of numbers.
    Get bonus from list length.
    Average the list.
    Add the bonus.
    return the total initiative value.'''
    if not len(speedGene):
        return 1
    temp = list()
    geneCount = int()
    for hexdecChar in speedGene:
        temp.append(int(hexdecChar, 16))
        geneCount += 1
    bonusInput = fetchBonus(geneCountMath(geneCount))
    return round(sum(temp) / geneCount) + bonusInput
 



 
### new mutation algorythm



def bootEnv(name):
    klass = EnvList(name)
    return klass()##### So, we're returning a called instance of whatever Environment we've chosen,
                        # but always without arguments...


EnvList = {
    'pig': Pig,
    'dice': Pig
}
validEnvNames = EnvList.keys()
# name_to_class = dict(some=SomeClass,
#                      other=OtherClass)



def pickEnv():
    trying = True
    while trying:
        try:
            choice = str(input("Which environment we runnin', Baby?"))
            ### check if response is in list of valid responses
            if choice in validEnvNames:
                print("A descision, made...\n\n")
                return choice
        except:
            print("That's not a valid entry at the moment.\nSorry if you believe this to be in error.\n\n\n\n\nbut it's not.")



        # gamestate will be filled with things like:
        # day or night, yes
        # player turn number, Sim can pass it into whatever's being passed to the Agent
        # round/turn/day number, yes, if it's a round based game, it can be handled by the game.
        # various point values, yes like...
        # table score/the pot, ez
        # all players total scores, list with length(all players)











waitingPlayers = list()# le HUH?


class Player:
    def __init__(self, id) -> None:
        self.ID = id
        pass

    def getPressed(self):
        return pygame.key.get_pressed()

    def queueUp(self):
        waitingPlayers.append(self.ID)


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
def mutateHexdec(gene:"str", radiationExposure:"float", radiationSeverity:"int"):
    '''raises/lowers the value of random bonds by severity modulo 16'''
    for i in range(len(gene)):
        if roll(1000, 994, 1000 * radiationExposure):### it shouldnt loop over from 0 to 15 and vice versa. Cap it.
            gene = replace_str_index(gene, i, HEX_OPTIONS[int(gene[i], 16) + random.choice([1 + radiationSeverity, 15 + radiationSeverity]) % 16])
    return gene

# What if the first output node( in sequence) firing means the network wants to update it's memory
# The second and third indicate location(, but by what formatting?).
# And the fourth output indicating what to set to.
# As for location formatting... From two independent floats to a point on a 2D grid?
##### Memory

class Agent:# Recieves rewards and observations, and returns an action
    def __init__(self, outputCount:"int"=1, geneSeq:'list[str]'=None, memory:"tuple(int, int)"=None, generation:"int"=1, radiation:"float"=0.01, toxicWaste:"float"=0.01):
        if memory:
            self.memories = list()
            for x in range(memory[0]):
                self.memories.append(str())
                for y in range(memory[1]):
                    self.memories[x] += "0"
        if geneSeq:# Build a NN from a Genome to handle the object.
            cleanGenes = list()
            self.speedGene = list()
            for i in range(len(geneSeq)):# extract speed
                temp = mutateHexdec(geneSeq[i], radiation, 0)## something to play with
                self.speedGene.append(temp[0])
                cleanGenes.append(temp[1:])
            del temp
            self.nn = NeuralNetwork(outputCount, cleanGenes, toxicWaste, generation)
            self.initiative = decodeSpeed("".join(self.speedGene))

        else:### No genome given, Check if any humans are waiting to play
            if not len(waitingPlayers):# if no players waiting, proceed
                ### say there were no users queued up, nor was a genome supplied,
                # Therefore an empty NN is being generated.
                pass
            pass### Make the object recieve input from input devices( wait on the User)
            # *This* Agent is a Player instead of a NN

    def seed(self):
        base = self.nn.seed()
        temp = list()
        for i in range(base):
            temp.append(f'\\n{self.speedGene[i]}{base[i]}')
        return temp




class Sim:
    def __init__(self, game:'str') -> None:
        '''Initialize an environment for this simulation'''
        self.environment = bootEnv(game)### set to a user defined class which imports from Env
        self.playerCount = self.environment.getPlayerCount()
        # self.envHistory = dict()## a place to store kept Environments by a name in str and list of settings?
        self.agents = list()# container for all Agents in the Sim
        self.initiativeOrder = list()

        # self.initiativeOrder.append(50, "LairAction")
    
    def sortF(self, incomingAgentID:"int"):
        return self.agents[incomingAgentID].initiative# range(1,20)

    def addAgent(self, agentID:"tuple(int, int)", environmentName:"str"):### load agent from genome into player dict, giving it a temporary 'system ID'. If it gets selected for reproduction, it will recieve a new ID and be saved.
        genome = loadGenome(agentID, environmentName)
        self.agents.append(Agent(# Generate Agent, store it in list with ID as index
            len(self.environment.actionOptions)-1),# how many output Nodes, -1 because any NN could always idle.
            genome)

        self.initiativeOrder.append(len(self.agents))# Put the AgentID in the initiative order list
        self.initiativeOrder.sort(key=self.sortF, reverse=True)# initOrder = list[agentID] sorted by self.agents[agentID].initiative
        



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






# # # Environments...
# An environment needs to be entirely self contained.
# The Sim will have asked an Agent what it wants to do.
# The Agent responds with an int representing it's selection.###
# update Env internal state with selected action from specified Agent.###
# Sim expects back an updated set of observations.###
 





# ______________________________________________________________________________________

# MAIN BLOCK
if __name__ == "__main__":
    pass

