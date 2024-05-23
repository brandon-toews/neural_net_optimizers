import numpy as np
from numba import njit


class Quantized_NN:
    def __init__(self, input_size, output_size, data_type=np.int8, criterion='MSE'):
        """
        Initialize a quantized neural network.

        Parameters:
        input_size (int): Number of input neurons.
        output_size (int): Number of output neurons.
        data_type (type): Data type for weights and biases (default: np.int8).
        criterion (str): Loss function to be used ('MSE' or 'CrossEntropy').
        """
        self.input_size = input_size
        self.output_size = output_size
        self.data_type = data_type
        self.layers = []
        self.weights = []
        self.biases = []
        self.train_losses = []
        self.train_accuracies = []
        self.optimizer = None
        self.criterion = criterion

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

    def add_hidden_layer(self, layer_type, neuron_size):
        """
        Add a hidden layer to the neural network.

        Parameters:
        layer_type (str): Type of activation function for the layer.
        neuron_size (int): Number of neurons in the hidden layer.
        """
        self.layers.append(self.activation_function[layer_type])
        if not self.weights:
            self.weights.append(np.zeros((self.input_size, neuron_size), self.data_type))
        else:
            next_layers_input_size = self.weights[-1].shape[1]
            self.weights.append(np.zeros((next_layers_input_size, neuron_size), self.data_type))
        self.biases.append(np.zeros(neuron_size, self.data_type))

    def add_output_layer(self, layer_type):
        """
        Add an output layer to the neural network.

        Parameters:
        layer_type (str): Type of activation function for the layer.
        """
        self.layers.append(self.activation_function[layer_type])
        if not self.weights:
            self.weights.append(np.zeros((self.input_size, self.output_size), self.data_type))
        else:
            output_layers_input_size = self.weights[-1].shape[1]
            self.weights.append(np.zeros((output_layers_input_size, self.output_size), self.data_type))
        self.biases.append(np.zeros(self.output_size, self.data_type))


    def calculate_loss(self, outputs, targets):
        """
        Calculate the loss using CrossEntropy.

        Parameters:
        outputs (np.ndarray): Predictions from the network.
        targets (np.ndarray): True labels.

        Returns:
        float: Calculated loss.
        """
        epsilon = 1e-7
        loss = -np.sum(targets * np.log(outputs + epsilon)) / targets.shape[0]
        return loss

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Parameters:
        x (np.ndarray): Input data.

        Returns:
        np.ndarray: Output after forward pass.
        """
        outputs = x
        for i in range(len(self.layers)):
            outputs = self.layers[i](np.dot(outputs, self.weights[i]) + self.biases[i])
        return outputs

    def flatten_weights(self):
        """
        Flatten the weight matrices into a single 1D array.

        Returns:
        np.ndarray: Flattened weights and biases.
        """
        flat_weights = np.concatenate([w.flatten() for w in self.weights] + [b.flatten() for b in self.biases])
        return flat_weights

    def reshape_weights(self, flat_weights):
        """
        Reshape a 1D array of weights back into the original weight matrices.

        Parameters:
        flat_weights (np.ndarray): Flattened weights and biases.

        Returns:
        tuple: Reshaped weights and biases.
        """
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
        """
        Get the flattened weights and biases.

        Returns:
        np.ndarray: Flattened weights and biases.
        """
        flat_parameters = self.flatten_weights()
        return flat_parameters

    def set_parameters(self, flat_parameters):
        """
        Set the weights and biases from flattened parameters.

        Parameters:
        flat_parameters (np.ndarray): Flattened weights and biases.
        """
        weights, biases = self.reshape_weights(flat_parameters)
        self.weights = weights
        self.biases = biases

    def fit(self, X, y, epochs):
        """
        Fit the neural network using the specified optimizer.

        Parameters:
        X (np.ndarray): Training data.
        y (np.ndarray): Training labels.
        epochs (int): Number of epochs to train.

        Returns:
        list: Training progress over epochs.
        """
        training_progress = None
        if self.optimizer is None:
            raise ValueError("Optimizer not set. Please set an optimizer before training the neural network.")
        else:
            training_progress = self.optimizer.train(X, y, epochs)
        return training_progress

    def predict(self, X):
        """
        Predict the output of the neural network.

        Parameters:
        X (np.ndarray): Input data.

        Returns:
        np.ndarray: Predicted output.
        """
        return self.forward(X)

    # Activation functions
    def sigmoid(self, x):
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
        return x * np.random.binomial(1, 1 - rate, size=x.shape)

    @staticmethod
    @njit
    def batch_norm(x, mean, var):
        return (x - mean) / np.sqrt(var + 1e-8)
