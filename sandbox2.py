
from numpy import transpose, random


def bin_to_float(b):
    """ Convert binary string to a float. """
    bf = int.to_bytes(int(b, 2), 8)  # 8 bytes needed for IEEE 754 binary64.
    return struct.unpack('>d', bf)[0]

# def int_to_bytes(n, length):  # Helper function
#     """ Int/long to byte string.

#         Python 3.2+ has a built-in int.to_bytes() method that could be used
#         instead, but the following works in earlier versions including 2.x.
#     """
#     return codecs.decode('%%0%dx' % (length << 1) % n, 'hex')[-length:]

def float_to_bin(value):  # For testing.
    """ Convert float to 64-bit binary string. """
    [d] = struct.unpack(">Q", struct.pack(">d", value))
    return '{:064b}'.format(d)


def diffDetect(a, b):
    '''can take itterable
    '''
    tempString = str()
    if len(a) < len(b):# if second item longer,
        a, b = b, a# swap 'em.
    for i in range(len(a)-len(b)):# buffer the temp string based on dif between item lengths.
        tempString += '0'
    for i in range(len(b)):
        if a[i] == b[i]:
            tempString += '0'
        else:
            tempString += '1'
    return tempString


# from numpy import random

# # so I need to build the synaptic weights( layers) from  the genes which Im also having trouble with...

# class Thing:
#     def __init__(self) -> None:
        
#         self.synaptic_weights = 2 * random.random((3, 1)) - 1

# thing1 = Thing()
# print(thing1.synaptic_weights)


# print(diffDetect('000000101010111100000100100011111011111111', '001010101111000001001000111110111111111111'))





# a = random.random((3, 1)) - 1
# print(a.transpose().transpose())




# # Define a hexadecimal string  

# # decode a hex string to a binary string
# gene = f'{0xABC123EFFF:0>42b}'
# print(int(gene, 2))



# print(int(101010111100000100100011111011111111))# failed





















# # # We train the neural network through a process of trial and error.
# # # Adjusting the synaptic weights each time.
# # def trainWeights(self, prediction_result, bestOutcome):
# #     '''Calculates the error in thinking and\n
# #     adjusts weights accordingly'''

# #     # Calculate the difference between the desired output and the expected output).
# #     error = bestOutcome - prediction_result

# #     # Multiply the error by the input and again by the gradient of the Sigmoid curve.
# #     # This means less confident weights are adjusted more.
# #     # This means inputs, which are zero, do not cause changes to the weights.
# #     adjustment = dot(prediction_result.T, error * self.__sigmoid_derivative(prediction_result))

# #     # Adjust the weights.
# #     self.synaptic_weights += adjustment