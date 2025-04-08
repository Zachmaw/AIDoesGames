### new mutation algorythm(s?)


### mutation parameters
# percentage of gene
# percentage of genome
# whether or not to delete/generate a gene
    # if yes, which?
    # and then if delete,
        # which gene?



 
# on an Agents initiative, 
# we pass to the Agent the current observations
# which we get from the Env.perspective method








# the NN is being stored in the init order list in the Sim class
# so I need a class function of Sim to call 
 




### Theres so much math involved with generation count...
# I want to proload Generation math which takes in how many generations you want to plan for and populates a list with that many of whatever the answer is...
# use txt files. set and reset the name to be the settings and range generated.






### when a pop is faced with mutation
# we need to know in what ways and how bad it is.
# A: number of flips per gene
# B: number of affected genes
# C: add/remove gene
    # Figure out add or del
        # A and B get normalized and compared?
    # then which gene?
        # if remove,  




















# import timeit
# import matplotlib.pyplot as plt
# import numpy as np
from numpy.random import choice, sample, randint
from commonFuncs import diceRoll
from math import ceil, exp, e, sin
import os










broCantType = True
while broCantType:
    try:
        geneLen = int(input("How many bits: "))
        geneCount = int(input("How many mutation iterations: "))
        broCantType = False
    except:
        print("Please, use numbers only.\nAnd NOT in word form...")
bstring = list()# generate bitstring with that length
for i in range(geneLen):
    bstring.append(str(randint(0, 2)))
thingy = bstring
print("".join(thingy))
for i in range(geneCount):# print that many variations of the first string with only one random bit flipped each
    second = list()
    whichBits = randint(0, geneLen)
    for i in range(geneLen):
        if not i in whichBits:
            second.append("0")
        else:
            second.append("1")
    print(thingy ^ "".join(second))










def make_filename(funcContext: list[str], param1: int, param2: int):### context = [funcName, param1Name, param2Name]
    folder = os.path.join("cached_data", funcContext)
    os.makedirs(folder, exist_ok=True)### constrain funcContext short and keep params numeric and padded to 4 deca-bits.
    return os.path.join(folder, f"{funcContext[0]}_{funcContext[1]}-{param1}_{funcContext[2]}-{param2}.txt")
# def get_filename(func_name: str, peak: int, max_len: int = 150):
#     return f"{func_name}_peak{peak}_len{max_len}.txt"





def rename_file(old_name: str, new_name: str):
    if os.path.exists(old_name):
        os.rename(old_name, new_name)
    else:
        print(f"File '{old_name}' does not exist.")

def write2file(filename: str, text: str):# one line at a time.
    with open(filename, 'a') as file:
        file.write(text + '\n')
def readFile(filename: str, dataType):
    try:
        with open(filename, 'r') as f:
            return [dataType(line.strip()) for line in f.readlines()]
    except FileNotFoundError:
        return False
    except:
        return False

def getFromBonusTable(peak: int, max_len: int = 150):
    filename = make_filename(peak, max_len)
    return generate_fetch_bonus_table(peak, max_len)





def newFetchBonus(geneLen: int, initiativeGeneCountPeak=25, max_len=150):
    values = getFromBonusTable(initiativeGeneCountPeak, max_len)
    return values[geneLen] if geneLen < len(values) else values[-1]










def decodeSpeed(hexdecSpeedGene:"str"):
    '''Alrighty... How do I want to do this?
    I want to make it so that the number 
    of characters in the speed gene affects
    the range of speeds available to that genome
    then I can just average out the hexdec digits
    by their index and transform the result with a
    coresponding speed list which has been cropped.
    How do I make a graph to indicate what I want...
    I need to know what slopes I want until what points.
    I need to know my x and y axiis.
    x axis = number of genes in the genome
    y axis = the 'bonus' to initiative value
    we can cap y at 0 and 1
    x is 0 to unbounded
    
    take the average, range(0,15)
    apply balance bonus to all, +3
    apply bonus based on gene count, +-2
    newRange = (1,20)'''
    
    pass


def generate_fetch_bonus_table(peak: int, max_len: int = 150):# pregenerate the lookup table for the simulation's given generation-based initialized genome peak.
    values = [round(((e/peak) * x * exp(-x/peak) * 6) - 3) for x in range(max_len)]## max_len is an index type of value.
    filename = make_filename(peak, max_len)
    with open(filename, 'w') as f:
        for val in values:
            f.write(str(val) + '\n')
    return values

def newFetchBonus(geneLen:"int", initiativeGeneCountPeak=25):
    

    ### I want to preload math. Which means we populate a list with 150 of whatever the answer is for a given peak value.
    # use txt files to save data. Use functions to set and reset the name to be the settings so I can load the data using the filename.
    # SETTINGS:
    # range to generate
    # peak value to use


    # what I have so far.
    return round(((e/initiativeGeneCountPeak) * geneLen * exp(-geneLen/initiativeGeneCountPeak) * 6) - 3)




def decodeInitiativeGene(speedGene:"list"):
    '''Takes the hexStr in after it's been gathered from the entire genome.
    Decode it to a list of numbers.
    Get a bonus from list length.
    Average the list values.
    Add the bonus.
    return the total initiative value.'''
    geneCount = len(speedGene)
    if not geneCount:# empty list. Genome had no genes to pull data from.
        return 1# default initiative score. Perfect balance.
    allValues = list()
    for hexdecChar in speedGene:
        allValues.append(int(hexdecChar, 16))
        geneCount += 1
    bonusInit = newFetchBonus(geneCount)
    return (sum(allValues) / geneCount) + bonusInit

 
############ GPT

def flipChance(value):
    """Assumes input is a float between 0 and 1"""
    scaled = value * 100
    flipped = 1000 - scaled
    return flipped

def mutate_gene_custom(gene: str, remainingMutations: int) -> str:
    gene_list = list(int(gene, 16))

    paramMutOdds = [
        0.05,  # source_type
        0.1,  # source_ID
        0.05,  # sink_type
        0.1,  # sink_ID
        0.6,  # weight x4 ### That's a lot of mutation chance. Make 0.7 the max for weights.
        0.02,  # sourceNodeBias
        0.08   # initiativeGene
    ]# mult by 100 and subtract from 1000, methinks.
    gene_list = list(gene)
    while remainingMutations:
        for i in range(len(gene_list)):
            if random() < paramMutOdds[i]:### make a bell or gamma or something function of x.
                # The mutation is applied to a nibble based on it's bounds within y.
                # Include a sine wave to make it so nibble_odds shift back and forth based on generation count.
                

                gene_list[i] = str((int(gene_list[i], 16) + choice([-1, 1])) % 16)
                remainingMutations -= 1
    return ''.join(gene_list)

def mutate_genome(parentGenome: list[str], mutRation: float, addRemoveGeneDifficulty: float, addRemovePreference: float) -> list[str]:### mutRation represents the percentage of nibbles in the genome to be mutated.
    '''returns the passed genome\nmutated by mutRation * 10'''### I need to be able to force a 100% mutation to a genome.
    indices_to_mutate = round(10 * len(parentGenome) * mutRation)# odds for coin dolage is based on calculated ratio from genome length and mutRation.
    new_genome = list()
    genomeMutPlan = list()# of ints with length number_of_genes_to_be_mutated
    genesDeling = list()# of indices of the genes we plan to delete.
    genesAdding = int()# how many fresh random genes to generate at the end.
    while indices_to_mutate:
        for i in range(len(parentGenome)):### if indicies_to_mutate/10 >= addRemoveGeneDifficulty: Roll to addRemove then if True: again to see if add or remove.
            if not i:### first iteration. Is there a faster way? or is the cost to check negligible?
                genomeMutPlan.append(int)
            if diceRoll(20, 16, 2):# roll to see if this gene gets selected to recieve a mutation point.
                indices_to_mutate -= 1
                genomeMutPlan[i] += 1

    for i in range(len(genomeMutPlan)):
        if genomeMutPlan[i]:
            ### if there are enough points to cross the threshhold, we spin the wheel to see if we delete this gene.
            if diceRoll(100, 51, 100-round(100*addRemoveGeneDifficulty)):
                if diceRoll(100, 1, round(100*addRemovePreference)):
                    genesAdding += 1### or just generate the gene?
                else:
                    # this one's going byebye, don't bother mutating. log it's index for deletion
                    genesDeling.append(i-len(genesDeling))# so when deleting, I can just go in order, right?
                    new_genome.append(parentGenome[i])### or can I just not add it to new_genome?
            else:# not adding or removing a gene, just mutating it.
                new_genome.append(mutate_gene_custom(parentGenome[i], genomeMutPlan[i]))# THEN take each gene, send it to the mutator with a random number of mutation coins up to len(gene).
        else:# no mutations allowed for the given gene
            new_genome.append(parentGenome[i])

    for igene in genesDeling:
        del new_genome[igene]
    for i in range(genesAdding):
        new_genome.append(randomOneGene())

    return new_genome




# max = gene count * 10
# mutRation is not chance of mutation, it's ratio thereof.
# mutRation * 100 = % of max to randomly select.
# we need to know exactly how many nibbles we're adjusting
# so we can decrement a running total.



def funcSine(generation):# f(x)=mx+A\sin(Bx) m=0.7, A=8, B=0.35 ### setting, What is y? mutationCount? something else?
    m = 0.7
    A = 8
    B = 0.35
    return m*generation+A*sin(B*generation)
def bitCombine(argA:"str", argB:"str"):# overlay mutation bitstring with gene bitstring
    temp = list()
    for i in range(len(argA)):
        temp.append(str((int(argA[i]) + int(argB[i])) % 2))
    return "".join(temp)
def binToHex(binaryString):
    return hex(int(binaryString, 2))
def randomOneGene():
    gene = list()
    for i in range(9):# speed's not random
        gene.append(int(choice(range(16)), 16))
    gene.append(int(randint(5, 12), 16))
    return "".join(gene)
def init_random_Genome(geneCount:"int"):
    genome = list()
    for i in range(geneCount):
        genome.append(randomOneGene())
    return genome# a list of hexdec strings( each with len(10))
# def mutateBitstring(bitstring:"str", bonus):
#     return bitCombine(bitstring, generateMutationBitstring(36, bonus))## 36??? Yes, 36.




#####
#####
temp = mutate_genome(parentGenome, radiation, newGeneChance)### mutation shenanagins
#####
##### ### Find out how harsh of radiation to apply based on return from mutation/generation sine function 




### every ten generations, 80% of the population should be cross-breeds of leading genomes

# What am I really trying to do?
# when the Sim is generating a population, it does so one NN at a time.
# To generate each NN, we take the parent genome( a list of hexStr) and,
# for each gene, apply our special mutation function( the gene, ) appending it to a new list( or do I have to?)).

# As for mutation:
    # Based on input odds, we want to force a given floor(given_percentage of len(genome)) to mutate.
    # When a gene has been selected for mutation, based on passed odds, 
    # we want the given number of randomly selected indicies in that gene to mutate,
    # but only one increment/decrement per nibble per generation.
        # This would mean the maximum number of mutations per gene is equal to it's length( 10 nibbles).
        # If only one nibble can mutate per gene per generation,
        # then the max number of mutations per generation per genome is 1000% of genome length?
    # We would like the odds of each index to have individual odds of selection, based on the parameters they encode for.
    # The encoded perameters per nibble are as follows.
    # [source_type, source_ID, sink_type, sink_ID, weight, weight, weight, weight, sourceNodeBias, initiativeGene]





# input, genome length
# output, initiative bonus








filename = "log.txt"
write2file("Testing...", filename)





make_filename("fetchBonus", 25, 150)
# → 'fetchBonus_p1-25_p2-150.txt'

make_filename("mutationOdds", 200, 16)### but we need to know 3 things. Which segments mutation odds, how big the batch, and x( which is?).
# → 'mutationOdds_p1-200_p2-16.txt'
