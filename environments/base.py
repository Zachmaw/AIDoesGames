# import pygame


# so the Sim class calls one of the Game classes( which all inherit from the Env class.), but the advanceTime function is called where exactly??
# The Simulation is the one who advances time in the environment by one step/turn when it has the responses from the Agents!


## what if the Env class could import from a variety of classes
# instead of Env being the parent class?
# I looked into variable class importing and found nothing. It's not worth my time.

# All environments are either turn based or time based
class Env:
    def __init__(self, id):# The id is given to us from the Simulation
        self.ready = False
        self.id = id

    def advanceTime(self, actions:'list[int]', envRules:'function'):
        '''
        should take in a list of actions and\n
        return a list of observations\n
        which comes directly from envRules\n
        based on the game being played\n
        Each Agent is always allowed one action per timestep.\n
        It is important to maintain the order of Agent actions.'''
        return envRules(actions)### make sure actions.len() can be varied between timesteps.(as long as len(actions) == len(agents))


class TurnBased(Env):
    def __init__(self, id) -> None:
        super().__init__(id)
        self.tookTurn = {}
        self.moves = {}


class TimeBased(Env):
    def __init__(self, id) -> None:
        super().__init__(id)


class GameObj(TurnBased):
    def __init__(self, id):### add the other atributes a game object would have
        super().__init__(id)
        self.maxPlayers = 7# This game has a player limit
        self.wins = {}# it also can be won.
        self.ties = 0# In case nobody could claim the game.

    def awaitPlayerConnections(self):
        trying = True
        while trying:
            try:
                playerCount = int(input('How many players?'))
                if playerCount > self.maxPlayers:
                    print(f"The number is too high! The highest number of players you can have in this environment is {self.maxPlayers}")
                    Exception("The number is too high!(playerCount above maxPlayers)")
                for i in playerCount:
                    self.tookTurn[f'Player{str(i)}'] = False
                    self.moves[f'Player{str(i)}'] = None
                    self.wins[f'Player{str(i)}'] = 0
                trying = False
            except:
                print('Please, try again.')
        temp2 = input('')
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