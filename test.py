import copy

import numpy as np
import torch.nn as tnn
import random


# Define a simple neural network class
class SimpleNN:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []
        self.weights = []
        self.biases = []
        # self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        # self.weights_hidden_output = np.random.rand(hidden_size, output_size)

    def add_hidden_layer(self, layer_type, neuron_size):
        if layer_type == "sigmoid":
            self.layers.append(self.sigmoid)
            if not self.weights:
                self.weights.append(np.zeros((self.input_size, neuron_size), np.int8))
            else:
                next_layers_input_size = self.weights[-1].shape[1]
                self.weights.append(np.zeros((next_layers_input_size, neuron_size), np.int8))

            self.biases.append(np.zeros(neuron_size, np.int8))


    def add_output_layer(self, layer_type):
        if layer_type == "sigmoid":
            self.layers.append(self.sigmoid)
            if not self.weights:
                self.weights.append(np.zeros((self.input_size, self.output_size), np.int8))
            else:
                next_layers_input_size = self.weights[-1].shape[1]
                self.weights.append(np.zeros((next_layers_input_size, self.output_size), np.int8))

            self.biases.append(np.zeros(self.output_size, np.int8))



    def forward(self, x):
        outputs = x
        #print(f'inputs: {outputs.shape}')
        for i in range(len(self.layers)):
            outputs = self.layers[i](np.dot(outputs, self.weights[i]) + self.biases[i])
            #print(f'Layer {i}: {outputs.shape}')
        #hidden_output = self.sigmoid()
        #final_output = self.sigmoid(np.dot(hidden_output, self.weights_hidden_output))
        return outputs

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def set_weights(self, new_parameters):
        self.weights = new_parameters[0]
        self.biases = new_parameters[1]
        #for i in range(len(self.weights)):
        #    self.weights[i] = new_weights[i]
        # if layer == 'hidden':
        #    self.weights_input_hidden = weights
        # elif layer == 'output':
        #    self.weights_hidden_output = weights

def generate_population(initial_individual, population_size, weight_range=2):
    low = np.int8(-weight_range/2)
    high = np.int8(weight_range/2)
    population = []
    for i in range(population_size):
        new_individual = ([], [])
        for weights, biases in zip(initial_individual[0], initial_individual[1]):
            layer_weight_shape = weights.shape
            layer_biases_shape = biases.shape
            new_individual[0].append(np.random.randint(low, high, (layer_weight_shape[0], layer_weight_shape[1]), np.int8))
            new_individual[1].append(np.zeros(layer_biases_shape, np.int8))

        population.append(new_individual)

    return population

def fitness_function(nn, X, y):
    predictions = nn.forward(X)
    #criterion = tnn.MSELoss()
    #loss = criterion(predictions, y)
    error = np.mean((predictions - y) ** 2)
    return 1 / (error + 1)  # Inverse of error to maximize fitness


def crossover(parent1, parent2):
    #crossover_point = np.random.randint(len(parent1[:, 0]))
    child1, child2 = ([], []), ([], [])
    for i in range(len(parent1[0])):
        # Correctly create an array with the shape of the parent's layer, filled with zeros
        child1[0].append(np.zeros(parent1[0][i].shape, np.int8))
        child1[1].append(np.zeros(parent1[1][i].shape, np.int8))
        child2[0].append(np.zeros(parent2[0][i].shape, np.int8))
        child2[1].append(np.zeros(parent2[1][i].shape, np.int8))
        if np.random.rand() < 0.5:
            child1[1][i] = parent1[1][i]
            child2[1][i] = parent2[1][i]
        else:
            child2[1][i] = parent1[1][i]
            child1[1][i] = parent2[1][i]
        for n in range(len(parent1[0][i])):
            if np.random.rand() < 0.5:
                child1[0][i][n] = parent1[0][i][n]
                child2[0][i][n] = parent2[0][i][n]
            else:
                child2[0][i][n] = parent1[0][i][n]
                child1[0][i][n] = parent2[0][i][n]

    #child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    #child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return child1, child2


'''def mutate(weights, mutation_rate=0.01):
    if np.random.rand() > mutation_rate:
        which_layer = np.random.randint(len(weights))
        which_neuron = np.random.randint(weights[which_layer].shape[0])
        which_weight = np.random.randint(weights[which_layer].shape[1])
        weights[which_layer][which_neuron, which_weight] = np.random.rand()
        #weights[which_layer] = np.random.rand(weights[which_layer].shape[0], weights[which_layer].shape[1])

    return weights'''


def mutate_adjust_weights(individual, mutation_rate=0.01, max_delta=10):
    """
    Mutates an individual's weights by slightly adjusting them.

    Parameters:
    - individual: The neural network weights to mutate.
    - mutation_rate: The probability of a weight being mutated.
    - max_delta: The maximum change that can be applied to a weight.

    Returns:
    - The mutated individual.
    """
    for which_layer in range(len(individual[0])):
        if np.random.rand() < mutation_rate:
            #low = -weight_range/2

            which_neuron = np.random.randint(individual[0][which_layer].shape[0])
            for which_weight in range(individual[0][which_layer].shape[1]):
                # Generate a small change, ensuring it's within the range [-max_delta, max_delta]
                delta = np.random.randint(-max_delta, max_delta, 1, np.int8)
                # Apply the change to the weight
                individual[0][which_layer][which_neuron, which_weight] += delta[0]
                # Optional: Clip the weights to ensure they remain within a desired range, e.g., [-1, 1]
                #individual[which_layer][which_neuron, which_weight] = np.clip(
                    #individual[which_layer][which_neuron, which_weight], -20, 20)

            which_bias = np.random.randint(individual[1][which_layer].shape)
            # Generate a small change, ensuring it's within the range [-max_delta, max_delta]
            delta = np.random.randint(-1, 2, 1, np.int8)
            individual[1][which_layer][which_bias] += delta
            individual[1][which_layer][which_bias] = np.clip(
                individual[1][which_layer][which_bias], -1, 1
            )
    return individual


# Genetic Algorithm parameters
population_size = 20
generations = 50
mutation_rate = 0.3

# Create initial population
input_size = 2
hidden_size = 20
output_size = 1
weight_range = 40

#population = [np.random.rand(input_size, hidden_size) for _ in range(population_size)]
#nn = SimpleNN(input_size, hidden_size, output_size)

nn = SimpleNN(input_size, output_size)
nn.add_hidden_layer("sigmoid", hidden_size)
nn.add_output_layer("sigmoid")

population = generate_population((nn.weights, nn.biases), population_size, weight_range)



# Example training data (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

best_solution = (-np.inf, None)

# Genetic Algorithm loop
for generation in range(generations):
    # Evaluate fitness of each individual
    fitness_scores = []
    for individual in population:
        nn.set_weights(individual)
        score = fitness_function(nn, X, y)
        if score > best_solution[0]:
            best_solution = (score, copy.deepcopy(individual))
        fitness_scores.append(score)

    # Sort indices based on fitness_scores in descending order
    sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)

    # Reorder population based on sorted indices
    sorted_population = [population[i] for i in sorted_indices]
    population = sorted_population[:population_size // 2]  # Select top 50%

    # population = roulette_wheel_selection(population, fitness_scores)


    # Create next generation
    new_population = []

    while len(new_population) < population_size:
        parent1, parent2 = population[np.random.randint(len(population)-1)], population[np.random.randint(len(population)-1)]
        child1, child2 = crossover(parent1, parent2)
        new_population.append(mutate_adjust_weights(child1, mutation_rate))
        new_population.append(mutate_adjust_weights(child2, mutation_rate))

    population = new_population
    best_fitness = max(fitness_scores)
    print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

# Test the best individual
best_individual = sorted_population[0]
nn.set_weights(best_solution[1])
predictions = nn.forward(X)


# Adjust numpy print options
np.set_printoptions(precision=4, suppress=True)

print("Predictions after genetic optimization:")
print(predictions)

#nn.set_weights(best_individual)
predictions = nn.forward(X)
# Adjust numpy print options
np.set_printoptions(precision=4, suppress=True)
print("Predictions after genetic optimization:")
print(predictions)
