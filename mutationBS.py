
# IMPORTS
from numpy.random import choice
from random import randint



def diceRoll(dice: int, difficultyClass: int, bonus: int =0):
    roll = randint(1, dice)
    if roll == dice:# if critical roll
        return True
    if roll == 1:# if natural 1
        return False
    if roll + bonus >= difficultyClass:
        return True
    else:
        return False

# determine how many indicies we're mutating
# cycle through genes, offering them a chance at a mutation until all tokens are claimed
# Spend the tokens on a mutation plan.
# excecute the plan.
def mutate_genome(parentGenome: list[str], rads: float, toxicWasteDunks: float, addRemoveGeneDifficulty: float, addRemoveGeneBias: float) -> list[str]:# mutRation represents the percentage of genes in the genome to be mutated.
    '''returns the passed genome\nmutated by mutRation as a percentage.'''
    indices_to_mutate = (round(toxicWasteDunks, 1) * 10) * round(len(parentGenome) * rads)# Whole number [0, 10] times percentage of genome length. => mutation tokens.
    genomeMutPlan = []#, of int, with length = number_of_genes_to_be_mutated
    while indices_to_mutate:
        for i in range(len(parentGenome)):
            if not i:# first iteration.
                genomeMutPlan.append(int)
            if diceRoll(20, 16, 2):# roll to see if this gene gets selected to recieve a mutation point.### double check these odds, but they look good at a glance.
                genomeMutPlan[i] += 1
                indices_to_mutate -= 1
            if not indices_to_mutate:
                break
    new_genome = []
    genesAdding = int()# how many fresh random genes to generate at the end.
    for i in range(len(genomeMutPlan)):# For each gene which needs mutation
        if genomeMutPlan[i]:# If there shall be at least one mutation on the indexed gene### what???
            if diceRoll(10, 30, (round(addRemoveGeneDifficulty, 1)*10)+genomeMutPlan[i]):# If the gene has enough coin... Roll to see if it spends it on a new gene/suicide, or just plain mutation.## This is probably fine.
                if diceRoll(100, 100, round(100*addRemoveGeneBias)):
                    new_genome.append(parentGenome[i])
                    genesAdding += 1
                genomeMutPlan[i] = 0
            else:# not adding or removing a gene, just mutating it.
                new_genome.append(mutateOneGene(parentGenome[i], genomeMutPlan[i], ))### THEN take each gene, send it to the mutator with it's number of mutation coins up to len(gene).
        else:# no mutations allowed for the given gene
            new_genome.append(parentGenome[i])
    for i in range(genesAdding):
        new_genome.append(randomOneGene())
    return new_genome

def mutateOneGene(a, b, c):
    pass

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


#'''Generation dictates the number of random genes to be generated in 'virgin' networks.'''
#init_random_Genome(ceil(funcSine(generation)))###

