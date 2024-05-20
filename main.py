import numpy as np
import neural_network as cust_nn
import quantized_nn as qnn
import custom_optimizers as cust_optims
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import matplotlib.pyplot as plt
import time



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
    hidden_size = 5
    output_size = 1
    normal_nn = cust_nn.NeuralNetwork(input_size, hidden_size, output_size)

    # Train the neural network
    epochs = 500
    learning_rate = 20

    # Start timing
    start_time = time.time()
    normal_nn_history = normal_nn.train(X, y, epochs, learning_rate)

    # End timing for normal neural network training
    normal_nn_training_time = time.time() - start_time
    print(f"Normal NN training time: {normal_nn_training_time} seconds")


    # Test the neural network
    output = normal_nn.forward(X)
    print("Predicted output:")
    print(output)

    # Define the quantized neural network
    input_size = 2
    hidden_size = 20
    output_size = 1
    q_nn = qnn.Quantized_NN(input_size, output_size)
    q_nn.add_hidden_layer('sigmoid', hidden_size)
    q_nn.add_output_layer('sigmoid')

    # Define the optimizer
    pop_size = 20
    mutation_rate = 0.3
    weight_range = 40
    q_nn.optimizer = cust_optims.GeneticAlgorithm(q_nn, pop_size, mutation_rate, weight_range)

    # Train quantized neural network
    epochs = 50

    # Reset start time for quantized neural network training
    start_time = time.time()

    q_nn_history = q_nn.fit(X, y, epochs)

    # End timing for quantized neural network training
    q_nn_training_time = time.time() - start_time
    print(f"Quantized NN training time: {q_nn_training_time} seconds")

    # Test quantized neural network
    output = q_nn.forward(X)
    print("Predicted output:")
    print(output)

    plt.figure(figsize=(10, 5))

    while not len(normal_nn_history) == len(q_nn_history):
        q_nn_history.append(q_nn_history[-1])

    # Plot the histories
    plt.plot(normal_nn_history, label='Normal Model Accuracy')
    plt.plot(q_nn_history, label='Quantized Model with GA Accuracy')

    # Add training times as text on the plot
    plt.text(0.5, 0.10, f'Normal NN training time: {normal_nn_training_time:.2f} seconds',
             transform=plt.gca().transAxes, fontsize=9, horizontalalignment='center')
    plt.text(0.5, 0.05, f'Quantized NN training time: {q_nn_training_time:.2f} seconds',
             transform=plt.gca().transAxes, fontsize=9, horizontalalignment='center')

    plt.title('Model Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

