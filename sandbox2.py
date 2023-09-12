
from numpy import transpose, random



def diffDetect(a, b):
    '''can take string or list
    '''
    temp = str()
    if len(a) < len(b):
        a, b = b, a
    for i in range(len(a)-len(b)):
        temp += '0'
    for i in range(len(b)):
        if a[i] == b[i]:
            temp += '0'
        else:
            temp += '1'
    return temp

# Define a hexadecimal string  

# decode a hex string to a binary string
# gene = f'{0xABC123EFFF:0>42b}'
# print(gene)

# from numpy import random

# # so I need to build the synaptic weights( layers) from  the genes which Im also having trouble with...

# class Thing:
#     def __init__(self) -> None:
        
#         self.synaptic_weights = 2 * random.random((3, 1)) - 1

# thing1 = Thing()
# print(thing1.synaptic_weights)


# print(diffDetect('000000101010111100000100100011111011111111', '001010101111000001001000111110111111111111'))




a = random.random((3, 1)) - 1
print(a.transpose().transpose())
