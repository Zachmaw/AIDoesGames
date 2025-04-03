# ask how many bits
# generate bitstring with that length
# ask how many mutation iterations
# print that many variations of the first string with only one random bit flipped each and converted to int
### Why is the output smiley faces?

from numpy import random
def strxor(str1, str2):# GPT did it
    """XOR two strings, truncating to the length of the shorter string."""
    size = min(len(str1), len(str2))
    return "".join(str(int(str1[i]) ^ int(str2[i])) for i in range(size))

broCantType = True
while broCantType:
    try:
        genomeLen = int(input("How many bits: "))
        popCount = int(input("How many mutation iterations: "))
        broCantType = False
    except:
        print("Please, use numbers only.\nAnd NOT in word form...")
## del broCantType
bstring = list()# generate bitstring with that length
for i in range(genomeLen):
    bstring.append(str(random.randint(0, 2)))
thingy = bstring
print("".join(thingy))
for i in range(popCount):# print that many variations of the first string with only one random bit flipped each
    second = list()
    whichBits = random.randint(0, genomeLen)
    for i in range(genomeLen):
        if not i in whichBits:
            second.append("0")
        else:
            second.append("1")
    print(strxor(thingy, "".join(second)))



# random chance to mutate, bad.
# randomly placed but predetermined mutation count, better.
# systematicly varied placement of mutations, awefull.




### when a pop is faced with mutation
# we need to know in what ways and how bad it is.
# number of flips per gene
# number of affected genes
# that it?