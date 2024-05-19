import numpy as np
import neural_network as cust_nn
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy




def main():
    # Input data (4 samples, 2 features each)
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    # Output data (4 samples, 1 output each)
    y = np.array([[0], [1], [1], [0]])

    # Define the neural network
    input_size = 2
    hidden_size = 20
    output_size = 1
    nn = cust_nn.NeuralNetwork(input_size, hidden_size, output_size)

    # Train the neural network
    epochs = 10000
    learning_rate = 0.1
    nn.train(X, y, epochs, learning_rate)

    # Test the neural network
    output = nn.forward(X)
    print("Predicted output:")
    print(output)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

