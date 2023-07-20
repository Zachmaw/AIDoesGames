# import pygame

class Env:
    def __init__(self) -> None:
        pass
    def advanceTime(self, vectors:'list[float]'):
        '''
        should take in a list of actions##### and
        return a list of observations
        based on the game being played
        '''
        pass




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