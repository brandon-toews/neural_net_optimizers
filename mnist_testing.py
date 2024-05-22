import numpy as np
import torch
from numba import njit
import neural_network as cust_nn
import quantized_nn as qnn
import custom_optimizers as cust_optims
import custom_pytorch_models as cust_py_nn
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define transformations for the training set, augmenting the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the training and test set
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

'''# Initialize the PyTorch model
model = cust_py_nn.Mnist_Model()
# criterion = nn.CrossEntropyLoss()  # This includes softmax
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# Start timing
start_time = time.time()
# Training the standard PyTorch model with Adam optimizer
model.train_model(train_loader, 3)
# End timing for normal neural network training
pyt_nn_training_time = time.time() - start_time
print(f"Pytorch Adam training time: {pyt_nn_training_time} seconds")
accuracy = model.evaluate_model(test_loader)
print(f'Standard PyTorch Model Accuracy: {accuracy}%')
model.plot_metrics()'''

# Define the parameters for the genetic algorithm optimizer
population_size = 5
mutation_rate = 0.1
weight_range = 1300

# Initialize the PyTorch model with the genetic algorithm optimizer
ga_pytorch_model = cust_py_nn.Mnist_GA_Model(population_size, mutation_rate, weight_range)
# Start timing
start_time = time.time()
# Training the pytorch model with genetic algorithm optimizer
ga_pytorch_model.train_model(train_loader, 10)
# End timing for GA neural network training
pyt_nn_training_time = time.time() - start_time
print(f"Pytorch GA training time: {pyt_nn_training_time} seconds")
accuracy = ga_pytorch_model.evaluate_model(test_loader)
print(f'Genetic Algorithm PyTorch Model Accuracy: {accuracy}%')
ga_pytorch_model.plot_metrics()


# Convert data loaders to numpy arrays
def dataloader_to_numpy(loader):
    data = []
    targets = []
    for inputs, labels in loader:
        # Ensure inputs are reshaped correctly if they're not already flat
        reshaped_inputs = inputs.view(inputs.size(0), -1).numpy()
        data.append(reshaped_inputs)
        targets.append(labels.numpy())
    return np.concatenate(data), np.concatenate(targets)


'''train_data, train_labels = dataloader_to_numpy(train_loader)
test_data, test_labels = dataloader_to_numpy(test_loader)

#save the data
np.save('train_data.npy', train_data)
np.save('train_labels.npy', train_labels)
np.save('test_data.npy', test_data)
np.save('test_labels.npy', test_labels)'''

# Load the data
train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')
test_data = np.load('test_data.npy')
test_labels = np.load('test_labels.npy')

# One-hot encode labels
train_labels_one_hot = np.eye(10)[train_labels]
test_labels_one_hot = np.eye(10)[test_labels]

# Define the parameters for the genetic algorithm optimizer
population_size = 20
mutation_rate = 0.1
weight_range = 60000

# Initialize quantized neural network with genetic algorithm optimizer
quantized_ga_model = qnn.Quantized_NN(28 * 28, 10, np.int8, 'CrossEntropy')
quantized_ga_model.add_hidden_layer('relu', 128)
quantized_ga_model.add_hidden_layer('relu', 64)
quantized_ga_model.add_output_layer('softmax')
quantized_ga_model.optimizer = cust_optims.GeneticAlgorithm(quantized_ga_model, population_size, mutation_rate, weight_range)

# Start timing
start_time = time.time()
# Training the quantized neural network with genetic algorithm optimizer
quantized_ga_model.fit(train_data, train_labels_one_hot, 20)
# End timing for GA neural network training
quantized_ga_training_time = time.time() - start_time
print(f"Quantized GA training time: {quantized_ga_training_time} seconds")

# Evaluate on test set
predictions = quantized_ga_model.forward(test_data)
test_loss = quantized_ga_model.calculate_loss(predictions, test_labels_one_hot)
print(f"Test Loss: {test_loss}")

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)
accuracy = np.mean(predicted_labels == test_labels) * 100
print(f"Test Accuracy: {accuracy}%")





