

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
    PlayerCount: int
    possibleActions: list[function]
    getObservations: function
    '''
    def __init__(self):
        self.ready = False
        self.init_order = list()
        self.thisRoundResponses = list()# to be filled to player count once an EnvChild inherits this.
        # self.gamestate = dict()
        self.PlayerCount = None
        self.possibleActions = list()
    
    def getObservations(self):
        pass### Error:Env not loaded

    def setRules(self, gameRules:"function"):
        self.rules = gameRules
    def actAndObserve(self, actionChoice:"int"):
        return self.rules(actionChoice)
    
    def getPlayerCount(self):
        return self.playerCount

    def advance(self):
        ### from whose turn it is in the game, update the internal game state based on actions taken
        # everyone whose turn it is not still recieves game data
        # but they also see it's not their turn.
        # This function is called after all players have submitted their responses.
        # somehow update the gameState from the response from player-whos-turn.
        # and set new observations for... all? just active player?
        #
        # Is it better to represent gamestate as a list of ints with variable length based on environment
        # would a list of int work?? does it have to be dict?
        # yeah, List is probably better. I can itterate over it.
        # I have to convert it to a list of floats anyway for the Agents.
        # I should find a way to just deliver the raw ints to Players.
        # but dict is easier for me to understand? keywise. as opposed to indexwise.
        ### so, from gamestate, get whose turn it is
        move = self.thisRoundResponses[self.gamestate["turnTracker"]]# Sim tracks whose turn it is.
        ### problem here is: roundOfResponses is ordered in sync with initOrder which is not the same as player order.
        if player == 0:
            self.p1Went = True
        else:
            self.p2Went = True


#####
class Chess(Env):
    def __init__(self, id, maxPlayers):### add the other atributes a game object would have( like??)
        super().__init__(id)
        self.playerLimit = maxPlayers# The player limit
        self.playersTookTurn = list()# list of bool with length = maxPlayers
        self.gamestate["turnTracker"] = int()# This game is turn based. int represents index in initOrder.
        # initOrder which has already been sorted so the highest speed has the lowest index. Descending order.

    def awaitPlayerConnections(self):### I don't think Env needs this.
        trying = True
        while trying:
            try:
                playerCount = int(input('How many players?'))
                if playerCount > self.playerLimit:
                    print(f"The number is too high! The highest number of players you can have in this environment is {self.playerLimit}")
                    Exception("The number is too high!(playerCount above maxPlayers)")
                for i in range(playerCount):### add each player to self.players
                    self.tookTurn[f'Player{str(i)}'] = False
                    self.moves[f'Player{str(i)}'] = None
                    self.wins[f'Player{str(i)}'] = 0
                trying = False
            except:
                print('Please, try again.\nJust for this game, ')### incomplete
        temp2 = input('')
        pass###

    def get_player_choice(self, p):### should be a function of Sim, not Env.
        """
        :param p: [0,1]
        :return: Move
        """
        return self.moves[p]


    def connected(self):### Again, Sim probably needs one of these, not Env.
        '''Check self.ready'''
        return self.ready

    def checkWinner(self):
        # cycle through all players answers, returning whose response is closest to the selected number without going over.
        # maybe track responses in a list, sort and return the top result
        p1 = self.moves[0].upper()[0]
        p2 = self.moves[1].upper()[0]

        winner = -1
        if p1 == "R" and p2 == "S":
            winner = 0
        elif p1 == "S" and p2 == "R":
            winner = 1
        elif p1 == "P" and p2 == "R":
            winner = 0
        elif p1 == "R" and p2 == "P":
            winner = 1
        elif p1 == "S" and p2 == "P":
            winner = 0
        elif p1 == "P" and p2 == "S":
            winner = 1
        return winner


class TurnBased(Env):
    def __init__(self, id):
        super().__init__(id)
        self.tookTurn = {}
        self.moves = {}
        ### and essentially just make it so if it's not that Agent's turn, it is blindfolded



    def resetWent(self):
        self.p1Went = False
        self.p2Went = False

    def allWent(self):### if not any(list of bool representing players)
        ### make variable to number of players
        return self.p1Went and self.p2Went















