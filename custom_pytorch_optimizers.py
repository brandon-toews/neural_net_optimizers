import torch
from torch.optim.optimizer import Optimizer
import numpy as np
import copy


class GeneticAlgorithmOptimizer(Optimizer):
    def __init__(self, model, population_size=20, mutation_rate=0.3, weight_range=40):
        self.params = [param for param in model.parameters()]
        defaults = dict(population_size=population_size, mutation_rate=mutation_rate,
                        weight_range=weight_range)
        super(GeneticAlgorithmOptimizer, self).__init__(self.params, defaults)

        self.model = model

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.weight_range = weight_range
        self.population = self.generate_population(self.params, population_size, weight_range)
        self.best_solution = (-np.inf, None)

    def _get_model_parameters(self):
        return [param.data.clone().detach().cpu().numpy() for param in self.model.parameters()]

    def _set_model_parameters(self, individual):
        for param, individual_param in zip(self.model.parameters(), individual):
            param.data = torch.tensor(individual_param, dtype=param.data.dtype, device=param.data.device)

    def generate_population(self, initial_individual, population_size, weight_range):
        low = -weight_range / 2
        high = weight_range / 2
        population = []
        for i in range(population_size):
            new_individual = [np.random.uniform(low, high, size=param.size()).astype(np.float32) for param in initial_individual]
            population.append(new_individual)
        return population

    def fitness_function(self, X, y):
        with torch.no_grad():
            outputs = self.model(X)
            loss = self.model.criterion(outputs, y)
        return 1 / (loss.item() + 1)

    def crossover(self, parent1, parent2):
        child1, child2 = [], []
        for layer1, layer2 in zip(parent1, parent2):
            shape = layer1.shape
            child1_layer = np.zeros(shape)
            child2_layer = np.zeros(shape)
            mask = np.random.rand(*shape) > 0.5
            child1_layer[mask] = layer1[mask]
            child1_layer[~mask] = layer2[~mask]
            child2_layer[mask] = layer2[mask]
            child2_layer[~mask] = layer1[~mask]
            child1.append(child1_layer)
            child2.append(child2_layer)
        return child1, child2

    def neuron_crossover(self, parent1, parent2):
        child1, child2 = [], []
        for layer1, layer2 in zip(parent1, parent2):
            shape = layer1.shape
            num_neurons = shape[1] if len(shape) > 1 else shape[0]  # Adjust for biases
            child1_layer = np.copy(layer1)
            child2_layer = np.copy(layer2)
            for i in range(num_neurons):
                if np.random.rand() < 0.5:
                    if len(shape) > 1:  # Weights
                        child1_layer[:, i] = layer2[:, i]
                        child2_layer[:, i] = layer1[:, i]
                    else:  # Biases
                        child1_layer[i] = layer2[i]
                        child2_layer[i] = layer1[i]
            child1.append(child1_layer)
            child2.append(child2_layer)
        return child1, child2

    def mutate(self, individual, max_delta=10):
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                delta = np.random.uniform(-max_delta, max_delta, individual[i].shape).astype(np.float32)
                individual[i] += delta
        return individual

    def step(self, closure=None):
        if closure is not None:
            closure()

        X, y, _ = closure()
        fitness_scores = []
        for individual in self.population:
            self._set_model_parameters(individual)
            score = self.fitness_function(X, y)
            fitness_scores.append(score)
            if score > self.best_solution[0]:
                self.best_solution = (score, copy.deepcopy(individual))

        sorted_indices = np.argsort(fitness_scores)[-self.population_size // 2:]
        sorted_population = [self.population[i] for i in sorted_indices]
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = sorted_population[np.random.choice(len(sorted_population))], sorted_population[np.random.choice(len(sorted_population))]
            # parent1, parent2 = np.random.choice(sorted_population, 2, replace=False)
            child1, child2 = self.neuron_crossover(parent1, parent2)
            new_population.append(self.mutate(child1))
            new_population.append(self.mutate(child2))
        self.population = new_population

        self._set_model_parameters(self.best_solution[1])
        return self.best_solution[0]




