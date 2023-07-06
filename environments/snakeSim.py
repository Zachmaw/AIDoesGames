import random

# defaultGenome = {}
# defaultGenome['Brain'] = list()
# defaultGenome['weight'] = int()
# defaultGenome['maxFood'] = int()

class Cell:
    def __init__(self) -> None:
        self.contains = 'Air'
    
class World:### n-D grid where each cell can contain either a piece of snake, a piece of food or neither.
    def __init__(self, size:'int') -> None:
        temp = list()
        for i in range(size):
            temp2 = list()
            for n in range(size):
                temp2.append(Cell())
            temp.append(temp2)
        self.composition = temp
        del temp
        del temp2
    def render(self):
        pass

class Snake:
    def __init__(self, genes:'dict', energy) -> None:
        self.food = 0
        self.age = 0
        self.length = 1
    def update(self):#####
        pass

def core():###
    pass

if __name__ == "__main__":
    core()
    print('Complete')