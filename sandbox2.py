
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