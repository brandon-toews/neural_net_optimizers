import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)

        # Initialize biases
        self.bias_hidden = np.random.rand(hidden_size)
        self.bias_output = np.random.rand(output_size)

    def forward(self, X):
        # Forward pass
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y, output, learning_rate):
        # Calculate error
        error = y - output
        accuracy = np.mean((output - y) ** 2)
        accuracy = 1 / (accuracy + 1)
        print(f'Accuracy: {accuracy}')

        # Calculate gradients for weights and biases
        d_output = error * sigmoid_derivative(output)
        error_hidden_layer = d_output.dot(self.weights_hidden_output.T)
        d_hidden = error_hidden_layer * sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * learning_rate
        self.bias_output += np.sum(d_output, axis=0) * learning_rate
        self.weights_input_hidden += X.T.dot(d_hidden) * learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0) * learning_rate

        return accuracy

    def train(self, X, y, epochs, learning_rate):
        training_progress = []
        for epoch in range(epochs):
            print(f'Epoch: {epoch} of {epochs}')
            output = self.forward(X)
            accuracy = self.backward(X, y, output, learning_rate)
            training_progress.append(accuracy)

        return training_progress


