import os.path


def saveGenome(genes, genomeID:"tuple(int, int)", envStr:"str"):
    with open(os.path.join(os.path.realpath(__file__), f"networks\\{envStr}\\NN{genomeID}.txt", "w")) as f:
        f.writelines(genes)# Sim.initOrder[currentInitiative].seed()

def loadGenome(genomeID:"tuple(int, int)", envStr:"str"):
    '''returns a list of hexdecimal strings\n
    but the first one is made of the first character from all of 'em cut off and stuck together\n
    (in order, of course).'''
    speed = list()
    genome = list()
    with open(os.path.join(os.path.realpath(__file__), '..', f"networks\\{envStr}\\NN{genomeID[0]}-{genomeID[1]}.txt"), "r") as f:
        for line in f.readlines():
            line2 = line.strip()
            speed.append(line2[0])
            genome.append(line2[1:])
    genome.insert(0, speed)
    return genome


for i in loadGenome((0,0), "ggez"):
    print(str(i))