from base import Env
from math import floor

"""
Date Modified:  Jun 20, 2023 - Current
Author: Zachmaw
With much code stolen from: Tech With Tim : flappy bird with NN.
"""
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
        for i in range(self.PlayerCount):
            self.players.append(int())
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
            'possibleActions': 'Greed or Chill'### how can a user idle? must be presented as an option...
        }
        
    def roll(self, d:'int'):
        return randint(0,d)
    def nextTurn(self):
        self.turn += 1# turn counter goes up
        self.turn = self.turn % self.PlayerCount# if it goes over, it resets
        self.tempScore = 0# pot resets

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
            observations.append([self.turn, self.roundNum, p, self.tempScore, self.leadScore])
            ### consider returning the general data and then the personal data
        ### was that really it, here?
        self.tempScore = 0# reset score for the next turn or roll
        return observations
    
    def getObservations(self, who:"int"):
        ### how can the Env know which player number is observing?
        pass### return

    # def advance(self, actions):
    #     return self.advanceTime(actions, self.ruleset)
    # def decodeActs(self, inputs:'tuple(float)'):# decode action from floats
    #     result = int()
    #     if inputs[0] < 0.55:
    #         if inputs[1] < 0.55:### IDLE
    #             result = 0
    #         else:# Neuron B active, Hold.
    #             result = 2### anything else I need?
    #     else:
    #         if inputs[1] < 0.55:# Neuron A active, Roll.
    #             result = 1
    #         else:# Both Neurons Active
    #             if self.roll(1):
    #                 result = 1
    #             else:
    #                 result = 2
    #     return result

