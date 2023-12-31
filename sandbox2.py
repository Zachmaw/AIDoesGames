










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