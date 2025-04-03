

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# so the Sim class calls one of the Game classes( which all inherit from the Env class.), but the advanceTime function is called where exactly??
# The Simulation is the one who advances time in the environment by one step/turn when it has the responses from the Agents!

# All environments start off initiative based, but TurnBased modifies it slightly...
# How? Well, let's start be explaining Initiative based order.
# First, the Environment is selected. The Env rules are loaded into the Env class or it's child.
# Then, from envRules, we know the expected NN outputs and player count and can populate the Sim with the NN/Agents and their speed.
# Sim
class Env:### Neither this not it's children can accept arguments.
    '''
    All inheritors must declare the following:
    PlayerCount: int, 
    possibleActions: list[function], 
    getObservations: function, 
    userInfo: list[str], 
    getUserFriendlyObservations: function
    '''
    ### if Agent is a User:
    #       getUserFriendlyObservations()
    def __init__(self):
        self.ready = False
        self.init_order = list()
        self.thisRoundResponses = list()# to be filled to player count once an EnvChild inherits this.
        # self.gamestate = dict()
        self.PlayerCount = None
    
    def getObservations(self):
        pass### Error:Env not loaded
    def getUserFriendlyObservations(self):
        pass### Error:Env not loaded

    def setRules(self, gameRules:"function"):
        self.rules = gameRules
    def actAndObserve(self, actionChoice:"int"):
        return self.rules(actionChoice)
    
    def getPlayerCount(self):
        return self.playerCount
    def playerInfo(self):
        return self.userInfo# all inheritors must declare
    def advance(self):
        ### from whose turn it is in the game, update the internal game state based on actions taken
        # for turn based Environments:
        # everyone whose turn it is not still recieves game data
        # but they also see whos turn it is and who they are.

        ### severely unsure about the below text...
        # Is it better to represent gamestate as a list of ints with variable length based on environment
        # would a list of int work?? does it have to be dict?
        # yeah, List is probably better. I can itterate over it.
        # I have to convert it to a list of floats anyway for the Agents.
        # I should find a way to just deliver the raw ints to Players.
        # but dict is easier for me to understand? keywise. as opposed to indexwise.
        ### so, from gamestate, get whose turn it is
        move = self.thisRoundResponses[self.gamestate["turnTracker"]]# Sim tracks whose turn it is.
        ### problem here is: roundOfResponses is ordered in sync with initOrder which is not the same as player order. how is that a problem?
        if player == 0:
            self.p1Went = True
        else:
            self.p2Went = True
