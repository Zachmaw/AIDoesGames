

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















