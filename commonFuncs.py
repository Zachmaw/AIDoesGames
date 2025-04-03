from numpy import random

def diceRoll(d, dc, bonus):
    r = random.randint(1, d+1)
    if r == d:# if critical roll
        return True
    if r == 1:# if natural 1
        return False
    if dc <= r + bonus:
        return True
    else:
        return False
def hextobin(hexaString):
  return bin(int(hexaString, 16))[2:].zfill(len(hexaString) * 4)