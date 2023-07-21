# import pygame


# so the Sim class calls one of the Game classes( which all inherit from the Env class.), but the advanceTime function is called where exactly??


## what if the Env class could import from a variety of classes
class Env:# all Games have this function which is called from the Sim class
    def advanceTime(self, actions:'list[int]', envRules:'function'):
        '''
        should take in a list of actions and\n
        return a list of observations\n
        which comes directly from envRules\n
        based on the game being played\n
        Each Agent is always allowed one action per timestep.\n
        It is important to maintain the order of Agent actions.'''
        return envRules(actions)### make sure actions.len() can be varied between timesteps.(as long as len(actions) == len(agents))




class GameObj:
    def __init__(self, id, playerCount):### add the other atributes a game object would have
        self.ready = False
        self.id = id
        self.playerCount = playerCount
        self.tookTurn = {}
        self.moves = {}
        self.wins = {}
        for i in playerCount:
            self.tookTurn[f'Player{str(i)}'] = False
            self.moves[f'Player{str(i)}'] = None
            self.wins[f'Player{str(i)}'] = 0
        self.ties = 0# In case nobody could claim the game.

    def awaitPlayerConnections(self):
        pass###

    def get_player_move(self, p):
        """
        :param p: [0,1]
        :return: Move
        """
        return self.moves[p]

    def play(self, player, move):
        self.moves[player] = move
        if player == 0:
            self.p1Went = True
        else:
            self.p2Went = True

    def connected(self):
        '''Check self.ready'''
        return self.ready

    def bothWent(self):
        return self.p1Went and self.p2Went

    def winner(self):

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

    def resetWent(self):
        self.p1Went = False
        self.p2Went = False

















# print(pygame.K_0)
# print(pygame.K_w)
# print(pygame.K_LEFT)
# print(pygame.K_SPACE)
# print(pygame.K_DOLLAR)