











### objects that need defenition
# GameObj
# Player
    # User
    # NeuNet
# Genome
    # Gene











class Genome:
    def __init__(self) -> None:
        pass

class Player:
    def __init__(self, isUser:'bool') -> None:
        if isUser:
            pass### Make the object recieve input from input devices
            
        else:
            pass### Build a NN from a Genome to handle the object.
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


























"""
Date Modified:  Jun 20, 2023 - Current
Author: Zachmaw
With much code stolen from: Tech With Tim : flappy bird.
"""
import random
import os
import neat
import pickle

gen = 0

class Game:
    """
    Class representing the Game object.
    """
    MIN_PLAYERS = 1
    MAX_PLAYERS = 4

    def __init__(self):
        """
        Initialize the object
        :param x: starting x pos (int)
        :param y: starting y pos (int)
        :return: None
        """
        pass






def fitnessFun(genomes, config):
    """
    runs the simulation of the current population of
    birds and sets their fitness based on the distance they
    reach in the game.
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
