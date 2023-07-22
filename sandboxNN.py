
from numpy import exp, array, random, dot, exp2, tanh
class NeuralNetwork():
    def __init__(self, inputCount, genes):### just make every extra action neuron above what the Environment can handle an automatic, but small, penalty.
        self.brain = list()
        ##### HERE


        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1
    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __tanh(self, x, deriv = False):
        if deriv == True:
            return (1 - (tanh(exp2(2) * x)))
          # return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        return tanh(x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def trainWeights(self, prediction_result, bestOutcome):
        '''Calculates the error in thinking and\n
        adjusts weights accordingly'''

        # Calculate the difference between the desired output and the expected output).
        error = bestOutcome - prediction_result

        # Multiply the error by the input and again by the gradient of the Sigmoid curve.
        # This means less confident weights are adjusted more.
        # This means inputs, which are zero, do not cause changes to the weights.
        adjustment = dot(prediction_result.T, error * self.__sigmoid_derivative(prediction_result))

        # Adjust the weights.
        self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single output neuron).
        ### I need to calculate the new state of each neuron in order
        # one layer at a time, [0, ...].
        for layerIter in range(self.brain):###
            for neuron in self.brain[layerIter]:
                tn = self.perceptron(inputs)
                if layerIter == len(self.brain):# Is output layer
                    result = self.__sigmoid(tn)
                else:
                    result = self.__tanh(tn)
        return result
    #forward pass
    def perceptron(self, x):
        return dot(x, self.synaptic_weights.T) + self.bias
################
    # class SigmoidNeuron:
    #   #intialization
    #   def __init__(self):
    #     self.w = None
    #     self.b = None
    #   #forward pass
    #   def perceptron(self, x):
    #     return np.dot(x, self.w.T) + self.b

    #   def sigmoid(self, x):
    #     return 1.0/(1.0 + np.exp(-x))
    #   #updating the gradients using mean squared error loss
    #   def grad_w_mse(self, x, y):
    #     y_pred = self.sigmoid(self.perceptron(x))
    #     return (y_pred - y) * y_pred * (1 - y_pred) * x

    #   def grad_b_mse(self, x, y):
    #     y_pred = self.sigmoid(self.perceptron(x))
    #     return (y_pred - y) * y_pred * (1 - y_pred)
    #   #updating the gradients using cross entropy loss
    #   def grad_w_ce(self, x, y):
    #     y_pred = self.sigmoid(self.perceptron(x))
    #     if y == 0:
    #       return y_pred * x
    #     elif y == 1:
    #       return -1 * (1 - y_pred) * x
    #     else:
    #       raise ValueError("y should be 0 or 1")

    #   def grad_b_ce(self, x, y):
    #     y_pred = self.sigmoid(self.perceptron(x))
    #     if y == 0:
    #       return y_pred
    #     elif y == 1:
    #       return -1 * (1 - y_pred)
    #     else:
    #       raise ValueError("y should be 0 or 1")
    #   #model fit method
    #   def fit(self, X, Y, epochs=1, learning_rate=1, initialise=True, loss_fn="mse", display_loss=False):

    #     # initialise w, b
    #     if initialise:
    #       self.w = np.random.randn(1, X.shape[1])
    #       self.b = 0

    #     if display_loss:
    #       loss = {}

    #     for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):
    #       dw = 0
    #       db = 0
    #       for x, y in zip(X, Y):
    #         if loss_fn == "mse":
    #           dw += self.grad_w_mse(x, y)
    #           db += self.grad_b_mse(x, y)
    #         elif loss_fn == "ce":
    #           dw += self.grad_w_ce(x, y)
    #           db += self.grad_b_ce(x, y)

    #       m = X.shape[1]
    #       self.w -= learning_rate * dw/m
    #       self.b -= learning_rate * db/m

    #       if display_loss:
    #         Y_pred = self.sigmoid(self.perceptron(X))
    #         if loss_fn == "mse":
    #           loss[i] = mean_squared_error(Y, Y_pred)
    #         elif loss_fn == "ce":
    #           loss[i] = log_loss(Y, Y_pred)

    #     if display_loss:
    #       plt.plot(loss.values())
    #       plt.xlabel('Epochs')
    #       if loss_fn == "mse":
    #         plt.ylabel('Mean Squared Error')
    #       elif loss_fn == "ce":
    #         plt.ylabel('Log Loss')
    #       plt.show()

    #   def predict(self, X):
    #     Y_pred = []
    #     for x in X:
    #       y_pred = self.sigmoid(self.perceptron(x))
    #       Y_pred.append(y_pred)
    #     return np.array(Y_pred)
###############################
if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    for i in range(10000):
        neural_network.trainWeights(neural_network.think(training_set_inputs), training_set_outputs)

    print(f"New synaptic weights after training: \n{neural_network.synaptic_weights}")

    # Test the neural network with a new situation.

    print(f"Correct answer: \n{training_set_outputs}")
    print(f"Final output: \n{neural_network.think(training_set_inputs)}")
    print(f"Considering new situation [1, 0, 0] -> ?: {neural_network.think(array([1, 0, 0]))}")
######

# I'm certain we cant fully solve the problem presented with only one neuron
