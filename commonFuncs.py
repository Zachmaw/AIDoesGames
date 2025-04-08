from numpy import random

def diceRoll(dice: int, difficultyClass: int, bonus: int =0):
    roll = random.randint(1, dice+1)
    if roll == dice:# if critical roll
        return True
    if roll == 1:# if natural 1
        return False
    if roll + bonus >= difficultyClass:
        return True
    else:
        return False
def hextobin(hexaString):
  return bin(int(hexaString, 16))[2:].zfill(len(hexaString) * 4)