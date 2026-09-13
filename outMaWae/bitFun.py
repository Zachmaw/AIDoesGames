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
attempts = 0
while broCantType:
    try:
        geneLen = int(input("How many bits: "))
        mutCount = int(input("How many mutation iterations: "))
        broCantType = False
    except:
        attempts += 1
        if attempts <=5:
            print("Please, use numbers only.\nAnd NOT in word form...")
        elif attempts <= 9:
            print("You better know how to read...")
        else:
            print("Alright, that's enough. You're done."*5)
            quit()
            
## del broCantType
bstring = list()# generate bitstring with that length
for _ in range(geneLen):
    bstring.append(str(random.randint(0, 2)))
thingy = bstring
print("".join(thingy))
for _ in range(mutCount):# print that many variations of the first string with only one random bit flipped each
    second = list()
    whichBits = random.randint(0, geneLen)
    for i in range(geneLen):
        if not i in whichBits:
            second.append("0")
        else:
            second.append("1")
    print(strxor(thingy, "".join(second)))



# random chance to mutate, bad.
# randomly placed but predetermined mutation count, better.
# systematicly varied placement of mutations, awful.




### when a pop is faced with mutation
# we need to know in what ways and how bad it is.
# number of flips per gene
# number of affected genes
# that it?