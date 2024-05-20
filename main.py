import numpy as np
import torch

import neural_network as cust_nn
import quantized_nn as qnn
import custom_optimizers as cust_optims
import custom_pytorch_models as cust_py_nn
import matplotlib.pyplot as plt
import time


# Plot comparison
def plot_comparison(models):
    plt.figure(figsize=(12, 5))

    for i, model in enumerate(models):
        epochs = range(1, len(model.train_losses) + 1)

        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, model.train_losses, label=f'Model {i + 1} Training Loss')

        # Plot training accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, model.train_accuracies, label=f'Model {i + 1} Training Accuracy')

    plt.subplot(1, 2, 1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy vs. Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

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

    # Adjust numpy print options
    np.set_printoptions(precision=4, suppress=True)

    # Test the neural network
    output = normal_nn.forward(X)
    print("Predicted output:")
    print(output)

    pyt_nn = cust_py_nn.XOR_Model(input_size, hidden_size, output_size)
    # Start timing
    start_time = time.time()
    pyt_nn.train_model(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32),
                       epochs)
    # End timing for normal neural network training
    pyt_nn_training_time = time.time() - start_time
    print(f"Pytorch NN training time: {pyt_nn_training_time} seconds")

    # Test the model
    pyt_nn.eval()
    with torch.no_grad():
        predictions = pyt_nn(torch.tensor(X, dtype=torch.float32))
        print('Predictions:')
        print(predictions)
        accuracy = pyt_nn.calculate_accuracy(predictions, torch.tensor(y, dtype=torch.float32))
        print(f'Accuracy: {accuracy}')
        print('Rounded Predictions:')
        print(torch.round(predictions))

    pyt_nn.plot_metrics()

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

