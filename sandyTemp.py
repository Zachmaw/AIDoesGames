
### combine these funcs
# use max()

def bitCombine(argA:"str", argB:"str"):# overlay mutation bitstring with gene bitstring
    temp = list()
    for i in range(len(argA)):
        temp.append(str((int(argA[i]) + int(argB[i])) % 2))
    return "".join(temp)

def strxor(str1, str2):# GPT did it...
    """XOR two strings, truncating to the length of the shorter string."""
    size = min(len(str1), len(str2))
    return "".join(str(int(str1[i]) ^ int(str2[i])) for i in range(size))



