# randomly generate a maze with the given number of dimentions(2-4(5?))
# The goal will always be to reach a corner Cell from the opposite Cell.



class Universe:
    '''Uhh... Reality.'''
    def __init__(self):
        self.contents = list()

class Cell:
    '''A piece of reality.'''
    def __init__(self, coords):
        self.contains = None
        self.coords = coords

        pass

def iterBatch(inp, subp):
    for i in inp:
        subp(i)

def addLayer(univers:'Universe'):
    for i in univers.contents### trash

# universe = [# 1 is universes( how many universes each containing seperate versions of realities)
#     [# 2 is realities
#         [# 3 is columns
#             [# 4 is ranks
#                 [], [], [], []# and 5 is files
#             ], [], [], []
#         ], [], [], []
#         ],
#     [[], [], [], []],
#     [[], [], [], []],
#     [[[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]]# so that each column represents the four 'realities' it spreads across
# ]
try:
    dimCount = int(input('How many dimentions(/axis(plural. Must be more than one.))?'))
    size = int(input('How long on a side?'))
except:
    print('You must input a number only. Not too large, I sugest not above 4 as with 5 Dimentions this results in 1024 Cells.')
    print('Surely you understand the implications. I will not be held liable for ANY damages.')
universe = Universe()
### if I know how many axis i'll need, wait...
for i in range(dimCount):### for each layer, multiply what's there by size and wrap it in a new list?
    pass### add another layer of dimentionality
    new = list()
    for x in range(size):
        new.append(universe.contents)
    # addLayer(universe)