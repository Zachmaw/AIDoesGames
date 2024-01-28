from Base import Env
from math import floor

"""
Date Modified:  Jun 20, 2023 - Current
Author: Zachmaw
With much code stolen from: Tech With Tim : flappy bird with NN.
"""
import os
import neat
from random import randint

class Pig(Env):
    """
    Pig with a twist and some penalty rules.
    """
    # needs to be entirely self contained, means it is 'unpowered'.
    # I ask it a question, giving it some power in the proccess.
    # and it returns a response. The Observations the NEXT Agent in order sees.
    #
    #
    #
    def __init__(self):
        self.PlayerCount = 7
        self.roundMultiplier = 2
        self.DICE = [2,4,6,8,10,12,20]# also represents max round count
        self.players = list()# of int representing that index's player's score
        self.turn = int()# who's turn is it rn?
        self.roundNum = int()# What round are we on?
        self.tempScore = int()# the pot
        self.leadScore = int()# the highScore
        self.possibleActions = [
            self.idle,
            self.greed,
            self.secure
        ]
        uiData = {
            'possibleActions': 'Greed or Chill'### how can a user idle?
        }
    def idle(self, who:'int', penaltyMultiplier:'int'):# no selection was offered
        if who == self.turn:# penalized for idling on your turn, otherwise nothing happens
            self.players[who] -= 1 * penaltyMultiplier
    def greed(self, who:'int', penaltyMultiplier:'int'):# first output node gave positive
        if not who == self.turn:
            self.players[who] -= 2 * penaltyMultiplier
        else:
            if not self.roundNum:
                diceNum = 0
            else:
                diceNum = floor(self.roundNum / self.roundMultiplier)
            rolled = self.roll(self.DICE[diceNum])#every even round use the next dice up
            if rolled == 0:
                self.nextTurn()### Im not sure that's all I had to do here...
            else:# didn't roll 0
                self.tempScore += rolled
            pass###
    def secure(self, who:'int', penaltyMultiplier:'int'):# second input from vector was True instead
        if not who == self.turn:
            self.players[who] -= 2 * penaltyMultiplier
        else:
            self.players[who] += self.tempScore
            self.nextTurn()

    def nextTurn(self):
        self.turn += 1
        self.turn = self.turn % self.PlayerCount
        self.tempScore = 0
    def addPlayer(self):# Not Agent. Let Sim handle that shturf.
        self.players.append(int())
        pass### anything else? i really am not so sure there is.
    def ruleset(self, inputV:'int', playerNum, penaltyMultiplier:'int'=1):### this func could probably ascend to the parent class.
        # I need to determine which action was taken by which player and react accordinly.
        self.possibleActions[inputV](playerNum, penaltyMultiplier)




        ### compile results and return a fresh set of observations to the Sim

        # based on the move [roll/hold/idle]
        # return observations (for Agent: list[oservations])
        # this game has the observations[turnCount:int, roundCount:int, grandScore:int, tempScore:int, leadScore:int, penalty:int]
        #
        ##### what format am I returning exactly? same observations to all? yeah. no.
        # everyone only sees their own score and the highest one, not all of them.
        # which means everyone gets a personalized observation? yeah....
        observations = list()### Rename
        self.leadScore = sorted(self.players, reverse=True)
        for p in self.players:# for player in self.players:### really think about it...
            observations.append([self.turn, self.roundNum, p, self.tempScore, self.leadScore, self.penalty])
            ### consider returning the general data and then the personal data
        ### was that really it, here?
        self.tempScore = 0# reset score for the next turn or roll
        return observations
    def advance(self, actions):
        return self.advanceTime(actions, self.ruleset)
    def roll(self, d:'int'):
        return randint(0,d)
    def decodeActs(self, inputs:'tuple(float)'):# decode action from floats
        result = int()
        if inputs[0] < 0.55:
            if inputs[1] < 0.55:### IDLE
                result = 0
            else:# Neuron B active, Hold.
                result = 2### anything else I need?
        else:
            if inputs[1] < 0.55:# Neuron A active, Roll.
                result = 1
            else:# Both Neurons Active
                if self.roll(1):
                    result = 1
                else:
                    result = 2
        return result




def fitnessFun(genomes, config):
    """
    runs the simulation of the current population of
    *!(@&$*&#@ and sets their fitness based on the #*@^%&$%&^ they
    reach in the *@^#%$.
    """
    global WIN, gen
    win = WIN
    gen += 1

    # start by creating lists holding the genome itself, the
    # neural network associated with the genome and the
    # bird object that uses that network to play
    nets = []
    solutions = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        solutions.append(Bird(230,350))
        ge.append(genome)

    score = 0

    run = True
    while run and len(solutions) > 0:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        pipe_ind = 0
        if len(solutions) > 0:
            if len(pipes) > 1 and solutions[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():  # determine whether to use the first or second
                pipe_ind = 1                                                                 # pipe on the screen for neural network input

        for x, bird in enumerate(solutions):  # give each bird a fitness of 0.1 for each frame it stays alive
            ge[x].fitness += 0.1
            bird.move()

            # send bird location, top pipe location and bottom pipe location and determine from network whether to jump or not
            output = nets[solutions.index(bird)].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5:  # we use a tanh activation function so result will be between -1 and 1. if over 0.5 jump
                bird.jump()

        base.move()

        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            # check for collision
            for bird in solutions:
                if pipe.collide(bird, win):
                    ge[solutions.index(bird)].fitness -= 1
                    nets.pop(solutions.index(bird))
                    ge.pop(solutions.index(bird))
                    solutions.pop(solutions.index(bird))

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            score += 1
            # can add this line to give more reward for passing through a pipe (not required)
            for genome in ge:
                genome.fitness += 5
            pipes.append(Pipe(WIN_WIDTH))

        for r in rem:
            pipes.remove(r)

        for bird in solutions:
            if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
                nets.pop(solutions.index(bird))
                ge.pop(solutions.index(bird))
                solutions.pop(solutions.index(bird))

        print(gamestate)

        # break if score gets large enough
        '''if score > 20:
            pickle.dump(nets[0],open("best.pickle", "wb"))
            break'''
    quit()

def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(fitnessFun, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
