import numpy as np
from numba import njit

# Define a custom neural network class to create a quantized neural network
class Quantized_NN:
    def __init__(self, input_size, output_size, data_type=np.int8, criterion='MSE'):
        # Initialize the input and output size
        self.input_size = input_size
        self.output_size = output_size
        self.data_type = data_type
        # Initialize the layers, weights, and biases
        self.layers = []
        self.weights = []
        self.biases = []

        self.train_losses = []
        self.train_accuracies = []

        # Initialize the optimizer
        self.optimizer = None

        self.criterion = criterion

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
            self.weights.append(np.zeros((self.input_size, neuron_size), self.data_type))
        # If there are weights, initialize the weights for the next layer
        else:
            # Get the size of the next layer's input
            next_layers_input_size = self.weights[-1].shape[1]
            # Initialize the weights for the next layer
            self.weights.append(np.zeros((next_layers_input_size, neuron_size), self.data_type))

        # Initialize the biases for the layer based on the neuron size
        self.biases.append(np.zeros(neuron_size, self.data_type))

    # Add an output layer to the neural network
    def add_output_layer(self, layer_type):
        # Append the activation function to the layers list
        self.layers.append(self.activation_function[layer_type])
        # If there are no weights, initialize the weights for the input layer
        if not self.weights:
            # Initialize the weights for the input layer
            self.weights.append(np.zeros((self.input_size, self.output_size), self.data_type))
        # If there are weights, initialize the weights for the next layer
        else:
            # Get the size of the output layer's input
            output_layers_input_size = self.weights[-1].shape[1]
            # Initialize the weights for the output layer
            self.weights.append(np.zeros((output_layers_input_size, self.output_size), self.data_type))
        # Initialize the biases for the output layer based on the output size
        self.biases.append(np.zeros(self.output_size, self.data_type))

    @staticmethod
    @njit
    def calculate_loss(outputs, targets):
        epsilon = 1e-7
        loss = -np.sum(targets * np.log(outputs + epsilon)) / targets.shape[0]
        return loss

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

    def flatten_weights(self):
        """Flatten the weight matrices into a single 1D array."""
        flat_weights = np.concatenate([w.flatten() for w in self.weights] + [b.flatten() for b in self.biases])
        return flat_weights

    def reshape_weights(self, flat_weights):
        """Reshape a 1D array of weights back into the original weight matrices."""
        weights, biases = [], []
        index = 0
        for w in self.weights:
            size = w.size
            weights.append(flat_weights[index:index + size].reshape(w.shape))
            index += size
        for b in self.biases:
            size = b.size
            biases.append(flat_weights[index:index + size].reshape(b.shape))
            index += size
        return weights, biases


    def get_parameters(self):
        # Return the flattened weights and biases
        flat_parameters = self.flatten_weights()
        return flat_parameters

    def set_parameters(self, flat_parameters):
        # Reshape and set the weights and biases from flattened parameters
        weights, biases = self.reshape_weights(flat_parameters)
        self.weights = weights
        self.biases = biases

    '''# Return the weights and biases of the neural network
    def get_parameters(self):
        return [self.weights, self.biases]

    # Set the weights and biases of the neural network
    def set_parameters(self, new_parameters):
        self.weights = new_parameters[0]
        self.biases = new_parameters[1]'''

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
    @njit
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    @njit
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    @staticmethod
    @njit
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    @njit
    def conv2d(x, kernel):
        return np.convolve(x, kernel, mode='valid')

    @staticmethod
    @njit
    def max_pooling(x, pool_size):
        return np.max(x.reshape(-1, pool_size), axis=1)

    @staticmethod
    @njit
    def avg_pooling(x, pool_size):
        return np.mean(x.reshape(-1, pool_size), axis=1)

    @staticmethod
    @njit
    def flatten(x):
        return x.flatten()

    @staticmethod
    @njit
    def dropout(x, rate):
        return x * np.random.binomial(1, 1-rate, size=x.shape)

    @staticmethod
    @njit
    def batch_norm(x, mean, var):
        return (x - mean) / np.sqrt(var + 1e-8)
