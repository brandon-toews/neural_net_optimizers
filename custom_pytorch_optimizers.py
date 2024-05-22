import torch
from torch.optim.optimizer import Optimizer
import numpy as np
import copy


class GeneticAlgorithm(Optimizer):
    def __init__(self, model, population_size=20, mutation_rate=0.3, weight_range=40):
        self.params = [param for param in model.parameters()]
        defaults = dict(population_size=population_size, mutation_rate=mutation_rate,
                        weight_range=weight_range)
        super(GeneticAlgorithm, self).__init__(self.params, defaults)

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

class ParticleSwarm(Optimizer):
    def __init__(self, model, weight_range=1300, num_particles=20, c1=2.0, c2=2.0, w=0.5):
        self.params = [param for param in model.parameters()]
        defaults = dict(num_particles=num_particles, c1=c1, c2=c2, w=w)
        super(ParticleSwarm, self).__init__(self.params, defaults)

        self.model = model
        self.weight_range = weight_range
        self.num_particles = num_particles
        self.c1 = c1
        self.c2 = c2
        self.w = [w for param in model.parameters()]
        self.particles = self.generate_particles(self.params, self.num_particles, self.weight_range)
        self.global_best_position = None
        self.global_best_fitness = float('inf')

    def _get_model_parameters(self):
        return [param.data.clone().detach().cpu().numpy() for param in self.model.parameters()]

    def _set_model_parameters(self, individual):
        for param, individual_param in zip(self.model.parameters(), individual):
            param.data = torch.tensor(individual_param, dtype=param.data.dtype, device=param.data.device)

    def generate_particles(self, params, num_particles, weight_range):
        particles = []
        for i in range(num_particles):
            new_particle = Particle(params, weight_range)
            particles.append(new_particle)
        return particles

    def objective_function(self, X, y):
        with torch.no_grad():
            outputs = self.model(X)
            loss = self.model.criterion(outputs, y)
        return loss.item()

    def step(self, closure=None):
        if closure is not None:
            closure()

        X, y, _ = closure()
        for particle in self.particles:
            # Update velocity
            r1, r2 = np.random.rand(2)
            cognitive_velocity = self.c1 * r1 * (particle.best_position - particle.position)
            social_velocity = self.c2 * r2 * (self.global_best_position - particle.position)
            particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity

            # Update position
            particle.position += particle.velocity

            self._set_model_parameters(particle.position)

            # Evaluate fitness
            current_fitness = self.objective_function(X, y)

            # Update personal best
            if current_fitness < particle.best_fitness:
                particle.best_fitness = current_fitness
                particle.best_position = particle.position.copy()

            # Update global best
            if current_fitness < self.global_best_fitness:
                self.global_best_fitness = current_fitness
                self.global_best_position = particle.position.copy()

        # Print the best solution found
        print("Best Position (Weights and Biases):", self.global_best_position)
        print("Best Fitness (Loss):", self.global_best_fitness)

        return self.global_best_fitness


class Particle:
    def __init__(self, params, weight_range):
        low = -weight_range / 2
        high = weight_range / 2
        self.position = [np.random.uniform(low, high, size=param.size()).astype(np.float32) for param in params]
        self.velocity = [np.random.uniform(low, high, size=param.size()).astype(np.float32) for param in params]
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

