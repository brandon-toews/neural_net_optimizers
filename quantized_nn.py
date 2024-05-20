import numpy as np
import custom_optimizers as cust_optims

# Define a custom neural network class to create a quantized neural network
class Quantized_NN:
    def __init__(self, input_size, output_size):
        # Initialize the input and output size
        self.input_size = input_size
        self.output_size = output_size
        # Initialize the layers, weights, and biases
        self.layers = []
        self.weights = []
        self.biases = []

        self.train_losses = []
        self.train_accuracies = []

        # Initialize the optimizer
        self.optimizer = None

        # Dictionary of activation functions
        self.activation_function = {
            'sigmoid': self.sigmoid,
            'relu': self.relu,
            'softmax': self.softmax,
            'tanh': self.tanh,
            'conv2d': self.conv2d,
            'max_pooling': self.max_pooling,
            'avg_pooling': self.avg_pooling,
            'flatten': self.flatten,
            'dropout': self.dropout,
            'batch_norm': self.batch_norm
        }

    # Add a hidden layer to the neural network
    def add_hidden_layer(self, layer_type, neuron_size):
        # Append the activation function to the layers list
        self.layers.append(self.activation_function[layer_type])
        # If there are no weights, initialize the weights for the input layer
        if not self.weights:
            # Initialize the weights for the input layer
            self.weights.append(np.zeros((self.input_size, neuron_size), np.int8))
        # If there are weights, initialize the weights for the next layer
        else:
            # Get the size of the next layer's input
            next_layers_input_size = self.weights[-1].shape[1]
            # Initialize the weights for the next layer
            self.weights.append(np.zeros((next_layers_input_size, neuron_size), np.int8))

        # Initialize the biases for the layer based on the neuron size
        self.biases.append(np.zeros(neuron_size, np.int8))

    # Add an output layer to the neural network
    def add_output_layer(self, layer_type):
        # Append the activation function to the layers list
        self.layers.append(self.activation_function[layer_type])
        # If there are no weights, initialize the weights for the input layer
        if not self.weights:
            # Initialize the weights for the input layer
            self.weights.append(np.zeros((self.input_size, self.output_size), np.int8))
        # If there are weights, initialize the weights for the next layer
        else:
            # Get the size of the output layer's input
            output_layers_input_size = self.weights[-1].shape[1]
            # Initialize the weights for the output layer
            self.weights.append(np.zeros((output_layers_input_size, self.output_size), np.int8))
        # Initialize the biases for the output layer based on the output size
        self.biases.append(np.zeros(self.output_size, np.int8))

    # Forward pass through the neural network
    def forward(self, x):
        # Initialize the outputs with the input
        outputs = x
        # Loop through the layers, calculate the outputs, and feed them to the next layer
        for i in range(len(self.layers)):
            # Calculate the outputs for each layer
            outputs = self.layers[i](np.dot(outputs, self.weights[i]) + self.biases[i])
        # Return the final outputs
        return outputs

    # Return the weights and biases of the neural network
    def get_parameters(self):
        return [self.weights, self.biases]

    # Set the weights and biases of the neural network
    def set_parameters(self, new_parameters):
        self.weights = new_parameters[0]
        self.biases = new_parameters[1]

    # Fit the neural network using the specified optimizer
    def fit(self, X, y, epochs):
        training_progress = None
        if self.optimizer is None:
            raise ValueError("Optimizer not set. Please set an optimizer before training the neural network.")
        else:
            training_progress = self.optimizer.train(X, y, epochs)
        return training_progress

    # Predict the output of the neural network
    def predict(self, X):
        return self.forward(X)

    # Activation functions
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def conv2d(x, kernel):
        return np.convolve(x, kernel, mode='valid')

    @staticmethod
    def max_pooling(x, pool_size):
        return np.max(x.reshape(-1, pool_size), axis=1)

    @staticmethod
    def avg_pooling(x, pool_size):
        return np.mean(x.reshape(-1, pool_size), axis=1)

    @staticmethod
    def flatten(x):
        return x.flatten()

    @staticmethod
    def dropout(x, rate):
        return x * np.random.binomial(1, 1-rate, size=x.shape)

    @staticmethod
    def batch_norm(x, mean, var):
        return (x - mean) / np.sqrt(var + 1e-8)
