### new mutation algorythm(s?)









# the NN is being stored in the init order list in the Sim class
# so I need a class function of Sim to call 
 




### Theres so much math involved with generation count...
# I want to preload Generation math which takes in how many generations you want to plan for and populates a list with that many int/float/bool of whatever the answer is...
# use txt files. set and reset the name to be the settings and range generated.






















# import timeit
# import matplotlib.pyplot as plt
# import numpy as np
from random import choices
from numpy.random import choice, randint
from numpy import linspace, cumsum
from commonFuncs import diceRoll
from math import exp, e, sin, pi
import os











################ FILE FUNCS
def get_filename(funcContext: list[str], params: list[int]):### context = [funcName, param1Name, param2Name]
    folder = os.path.join("lookupTables", funcContext)
    os.makedirs(folder, exist_ok=True)### constrain funcContext short and keep params numeric and padded to 4 places
    return os.path.join(folder, f"{funcContext[0]}_{funcContext[1]}-{params[0]}_{funcContext[2]}-{params[1]}.txt")# file name format. name_data1-nums_data2-nums.txt
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






############### INITIATIVE RELATED FUNCS

def decodeInitiativeGene(speedGene:"list"):
    '''Takes the hexStr in after it's been gathered from the entire genome.
    Decode it to a list of numbers.
    Get a bonus from list length.
    Average the list values.
    Add the bonus.
    return the total initiative value.'''
    geneCount = len(speedGene)
    if not geneCount:# empty list. Genome had no genes to pull data from.
        return 1# default initiative score. Perfect balance.### wouldn't that be 0?
    allValues = list()
    for hexdecChar in speedGene:
        allValues.append(int(hexdecChar, 16))
        geneCount += 1
    bonusInit = newFetchBonus(geneCount)
    return (sum(allValues) / geneCount) + bonusInit



def generate_fetch_bonus_table(peak: int, max_len: int = 150):# pregenerate the lookup table for the simulation's given generation-based initialized genome peak.
    values = [round(((e/peak) * x * exp(-x/peak) * 6) - 3) for x in range(max_len)]## max_len is an index type of value.
    filename = get_filename(peak, max_len)
    with open(filename, 'w') as f:
        for val in values:
            f.write(str(val) + '\n')
    return values















 
############ GPT AND MUTATION



# what am I trying to do?
# based on generation count and a sine wave,
# apply a scalar to the entire list of base_odds,
# not just passing in the one we need.
# The scalar is derived from the sine func.
# THEN generate a random int from 0 to 9
# but with odds adjusted based on the scaled_odds.


paramMutOdds = [# requires normalization
    2,  # source_type
    5,  # source_ID
    2,  # sink_type
    5,  # sink_ID
    8, 8,# weight x2
    3,  # sourceNodeBias
    1   # initiativeGene
]# Represents odds compared to each other that a given nibble is selected for the distributed mutation
# I want these odds to shift with generation count according to a sine wave.
# even inicies shift up while odd indicies shift down.






def mutateOneGene(gene, num_mutations, mutOdds):
    gene_list = list(gene)
    while num_mutations > 0:
        for i in range(len(gene_list)):
            if random() < mutOdds[i]:
                gene_list[i] = str((int(gene_list[i], 16) + choice([-1, 1])) % 16)
                num_mutations -= 1
                if num_mutations == 0:
                    break
    return ''.join(gene_list)

def mutateOneGene(gene: str, num_mutations: int, genCount: int) -> str:
    gene_list = list(gene)

    # Modulate odds with generation-based scalar
    scalar_LTable = get_or_generate(modulate_param_mut_odds, ["mutOdds", ], [current_scalar], 15, 1.5)# paramMutOdds = get_or_generate(modulate_param_mut_odds, ["paramMutOdds"], [current_scalar])

    # scalar = get_param_mut_scalars(generation)
    mutMutOdds = [min(1.0, chances * scalar_LTable[genCount]) for chances in paramMutOdds]
    indexes = choices(len(paramMutOdds), paramMutOdds, k=num_mutations)
    ### make a sine function of generation.
    for i in range(len(indexes)):
        if randint() < mutMutOdds[i]:
            gene_list[i] = str((int(gene_list[i], 16) + choice([-1, 1])) % 16)
            num_mutations -= 1
            # if not howMuch:
            break

    return ''.join(gene_list)



                # The mutation is applied to a nibble based on it's bounds within y.
                # Include a sine wave to make it so nibble_odds shift back and forth based on generation count.











### np.sin(np.cumsum(1 + 0.5 * np.sin(x)) * (x[1] - x[0]))# accumulate frequency shift. my special line







#####
#####
new_genome.append(mutateOneGene(parentGenome[i], genomeMutPlan[i], ))### THEN take each gene, send it to the mutator with it's number of mutation coins up to len(gene).
# temp = mutate_genome(parentGenome, radiation, addRemovePreference, addRemoveGeneBias)### mutation shenanagins (parentGenome: list[str], radiation:, addRemovePreference)
#####
##### ### Find out how harsh of radiation to apply based on return from mutation/generation sine function 




### every f generations, 80% of the population should be cross-breeds of leading genomes

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
write2file(filename, "Testing...")





get_filename("fetchBonus", 25, 150)
# → 'fetchBonus_p1-25_p2-150.txt'

get_filename("mutationOdds", 200, 16)### but we need to know 3 things. Which segments mutation odds, how big the batch, and x( which is?).
# → 'mutationOdds_p1-200_p2-16.txt'






































def get_or_generate(usedFunction: FunctionType, function_context: list[str], params: list[int]) -> list[float]:
    filename = get_filename(function_context, params)
    if os.path.exists(filename):# print(f"Loading data from {filename}")
        return readFile(filename)
    else:# print(f"Generating new data for {filename}")
        data = usedFunction(params)
        write2file(filename, data)
        return data



def get_mod_scalar(generation_count):
    return generation_count + 15 % 30

mod_scalar = get_mod_scalar(generation_count)
paramMutOdds = get_or_generate(modulate_param_mut_odds, ["paramMutOdds"], [round(mod_scalar, 3)])


# I just want to know, based on current generation, how should the mutOdds be scaled
def get_param_mut_scalars(genCount: int):
    """
    Returns a list of scaling factors for parameter mutation odds based on generation count.
    A sine-wave-based modulation bumps mutation odds every 15 generations.
    """
    # Rescale x to make high peaks 15 generations apart (period = 30)
    x = linspace(genCount - 1, genCount, 1000)
    freq_wave = 1 + (0.5 * sin((2 * pi * x) / 30))  # period = 30 gens### plot that, make sure I don't need to chop that.
    phase = cumsum(freq_wave) * (x[1] - x[0])
    scalar = (sin(phase) + 1) / 2  # normalize [-1, 1] → [0, 1]
    return scalar

def modulate_param_mut_odds(params):# generate a modulated odds list based on params

    base_odds = [0.05, 0.1, 0.05, 0.1, 0.05, 0.2, 0.05, 0.2, 0.05, 0.15]### make settings-able
    mod_scalar = params[0]  # assume params[0] is your mod scalar [0, 1]
    modulated = [
        val * (0.5 + mod_scalar * 0.5) if i % 2 == 0
        else val * (1 - mod_scalar * 0.5)
        for i, val in enumerate(base_odds)
    ]
    return modulated

def modulate_param_mut_odds(params):
    generation_scalar = params[0]
    base_odds = [0.05, 0.1, 0.05, 0.1, 0.05, 0.2, 0.05, 0.2, 0.05, 0.15]
    
    modulated = [
        val * (0.5 + generation_scalar * 0.5) if i % 2 == 0
        else val * (1 - generation_scalar * 0.5)
        for i, val in enumerate(base_odds)
    ]
    return modulated





def genGenesPerGentin(generation: int, rate: float =0.7):# f(x)=mx+A\sin(Bx) m=0.7, A=8, B=0.35 ### setting, What is y? mutationCount? something else?
    frequency = 8
    amp = 0.35
    return rate*generation+frequency*sin(amp*generation)





def doubleSine(generationsPerHalfCycle: int):
    sin(cumsum(1 + 0.5 * sin(genCount)) * (genCount[1] - genCount[0]))# accumulate frequency shift# my special red line


    wave = (sin(gen * frequency) + 1) / 2  # scales to [0, 1]




# def getFromBonusTable(peak: int, max_len: int = 150):
#     filename = make_filename(peak, max_len)
#     return generate_fetch_bonus_table(peak, max_len)



def newFetchBonus(geneLen: int, initiativeGeneCountPeak=25, max_len=150):
    values = getFromBonusTable(initiativeGeneCountPeak, max_len)
    return values[geneLen] if geneLen < len(values) else values[-1]


def newFetchBonus(geneLen:"int", initiativeGeneCountPeak=25):
    ### I want to preload math. Which means we populate a list with 150 of whatever the answer is for a given peak value.
    # use txt files to save data. Use functions to set and reset the name to be the settings so I can load the data using the filename.
    # SETTINGS:
    # range to generate
    # peak value to use


    # what I have so far.
    return round(((e/initiativeGeneCountPeak) * geneLen * exp(-geneLen/initiativeGeneCountPeak) * 6) - 3)





