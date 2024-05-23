import torch
from torch.optim.optimizer import Optimizer
import numpy as np
import copy
from numba import njit


# Custom PyTorch Genetic Algorithm Optimizer
class GeneticAlgorithm(Optimizer):
    def __init__(self, model, population_size=20, mutation_rate=0.3, weight_range=40):
        """ Genetic Algorithm Optimizer for PyTorch models
        Args:
            model: PyTorch model to be optimized
            population_size (int): Number of individuals in the population
            mutation_rate (float): Probability of mutation for each individual
            weight_range (int): Range of the initial weights for the population """
        # Get the model parameters and store them as a list in optimizer
        self.params = [param for param in model.parameters()]
        # Initialize the optimizer with the parameters and defaults
        defaults = dict(population_size=population_size, mutation_rate=mutation_rate,
                        weight_range=weight_range)
        super(GeneticAlgorithm, self).__init__(self.params, defaults)

        # Store the model
        self.model = model

        # Store the parameters
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.weight_range = weight_range

        # Initialize the population
        self.population = self.generate_population(self.params, population_size, weight_range)
        # Initialize the best solution
        self.best_solution = (-np.inf, None)

    # Get the model parameters
    def _get_model_parameters(self):
        """ Get the model parameters
        Returns:
            List: List of numpy arrays containing the model parameters """
        # Return the model parameters as a list of numpy arrays
        return [param.data.clone().detach().cpu().numpy() for param in self.model.parameters()]

    # Set the model parameters
    def _set_model_parameters(self, individual):
        """ Set the model parameters
        Args:
            individual (List): List of numpy arrays containing the model parameters """
        # Set the model parameters from a list of numpy arrays
        for param, individual_param in zip(self.model.parameters(), individual):
            param.data = torch.tensor(individual_param, dtype=param.data.dtype, device=param.data.device)

    # Generate the initial population
    def generate_population(self, initial_individual, population_size, weight_range):
        """ Generate the initial population
        Args:
            initial_individual (List): Initial model parameters
            population_size (int): Number of individuals in the population
            weight_range (int): Range of the initial weights for the population
        Returns:
            population (List): List of individuals in the population """
        # Generate a population of random individuals within the weight range,
        # helpful to be able to define the initial search space
        low = -weight_range / 2
        high = weight_range / 2
        population = []
        # Generate random individuals within the weight range
        for i in range(population_size):
            # Generate a new individual with random weights and biases
            new_individual = [np.random.uniform(low, high, size=param.size()).astype(np.float32)
                              for param in initial_individual]
            # Append the new individual to the population
            population.append(new_individual)
        # Return the population
        return population

    # Fitness function
    def fitness_function(self, X, y):
        """ Fitness function for the genetic algorithm
        Args:
            X (Tensor): Input data
            y (Tensor): Target data
            Returns:
                Fitness score (float): Inverse of the loss for the model """
        # No gradient computation is needed
        with torch.no_grad():
            # Feed the input data to the model
            outputs = self.model(X)
            # Calculate the loss
            loss = self.model.criterion(outputs, y)
        # Return the fitness score as the inverse of the loss
        return 1 / (loss.item() + 1)

    # Layer level crossover function
    def layer_crossover(self, parent1, parent2):
        """ Layer level crossover function - Uniform crossover
        Args:
            parent1 (List): First parent individual
            parent2 (List): Second parent individual
        Returns:
            child1 (List): First child individual
            child2 (List): Second child individual """
        # Initialize the children
        child1, child2 = [], []
        # Loop through the layers of the parents
        for layer1, layer2 in zip(parent1, parent2):
            # Get the shape of the layer
            shape = layer1.shape
            # Initialize the child layers
            child1_layer = np.zeros(shape)
            child2_layer = np.zeros(shape)
            # Generate a random mask for the crossover
            mask = np.random.rand(*shape) > 0.5
            # Perform the crossover based on the mask
            child1_layer[mask] = layer1[mask]
            child1_layer[~mask] = layer2[~mask]
            child2_layer[mask] = layer2[mask]
            child2_layer[~mask] = layer1[~mask]
            # Append the child layers to the children
            child1.append(child1_layer)
            child2.append(child2_layer)
        # Return the children
        return child1, child2

    # Neuron level crossover function
    def neuron_crossover(self, parent1, parent2):
        """ Neuron level crossover function - Uniform crossover
        Args:
            parent1 (List): First parent individual
            parent2 (List): Second parent individual
        Returns:
            child1 (List): First child individual
            child2 (List): Second child individual """
        # Initialize the children
        child1, child2 = [], []
        # Loop through the layers of the parents
        for layer1, layer2 in zip(parent1, parent2):
            # Get the shape of the layer
            shape = layer1.shape
            # Get the number of neurons in the layer
            num_neurons = shape[1] if len(shape) > 1 else shape[0]  # Adjust for biases
            # Initialize the child layers
            child1_layer = np.copy(layer1)
            child2_layer = np.copy(layer2)
            # Perform the neuron level crossover
            # Loop through the neurons
            for i in range(num_neurons):
                # Perform the crossover with a 50% probability
                if np.random.rand() < 0.5:
                    if len(shape) > 1:  # Weights
                        # Swap the weights of the neurons
                        child1_layer[:, i] = layer2[:, i]
                        child2_layer[:, i] = layer1[:, i]
                    else:  # Biases
                        # Swap the biases
                        child1_layer[i] = layer2[i]
                        child2_layer[i] = layer1[i]
            # Append the child layers to the children
            child1.append(child1_layer)
            child2.append(child2_layer)
        # Return the children
        return child1, child2

    # Mutation function
    def mutate(self, individual, max_delta=10):
        """ Mutation function for the genetic algorithm
        Args:
            individual (List): Individual to be mutated
            max_delta (int): Maximum delta for mutation
        Returns:
            child (List): Mutated individual """
        # Perform mutation with the mutation rate probability
        if np.random.rand() < self.mutation_rate:
            # Loop through the layers of the individual
            for i in range(len(individual)):
                # Generate random delta for mutation within the maximum delta
                delta = np.random.uniform(-max_delta, max_delta, individual[i].shape).astype(np.float32)
                # Perform mutation
                individual[i] += delta
        # Return the mutated individual
        return individual

    # Step function
    def step(self, closure=None):
        """ Step function for the genetic algorithm
        Args:
            closure (callable): A closure that reevaluates the model and returns the loss
        Returns:
            Best solution fitness score (float) """
        # If closure is provided, reevaluate the model
        if closure is not None:
            closure()

        # Get the input data and target data from the closure
        X, y, _ = closure()
        # Initialize the fitness scores
        fitness_scores = []
        # Loop through the population
        for individual in self.population:
            # Set the model parameters to the individual
            self._set_model_parameters(individual)
            # Calculate the fitness score for the individual
            score = self.fitness_function(X, y)
            # Append the fitness score to the list
            fitness_scores.append(score)
            # Update the best solution
            if score > self.best_solution[0]:
                self.best_solution = (score, copy.deepcopy(individual))

        # Sort the population based on the fitness scores, select the top best half indices
        sorted_indices = np.argsort(fitness_scores)[-self.population_size // 2:]
        # Select the top best half of the population using the sorted indices
        sorted_population = [self.population[i] for i in sorted_indices]
        # Initialize the new population
        new_population = []
        # Loop until the new population reaches the desired size
        while len(new_population) < self.population_size:
            # Select two parents randomly from the sorted population
            parent1, parent2 = (sorted_population[np.random.choice(len(sorted_population))],
                                sorted_population[np.random.choice(len(sorted_population))])
            # Perform the neuron level crossover to generate two children
            child1, child2 = self.neuron_crossover(parent1, parent2)
            # Mutate the children and append them to the new population
            new_population.append(self.mutate(child1))
            new_population.append(self.mutate(child2))
        # Update the population with the new population
        self.population = new_population
        # Set model parameters to the best solution
        self._set_model_parameters(self.best_solution[1])
        # Return the best solution fitness score
        return self.best_solution[0]


# Custom PyTorch Particle Swarm Optimizer
class ParticleSwarm(Optimizer):
    def __init__(self, model, weight_range=1, num_particles=20, clipping=False):
        """ Particle Swarm Optimizer for PyTorch models
        Args:
            model: PyTorch model to be optimized
            weight_range (int): Range of the initial weights for the particles
            num_particles (int): Number of particles in the swarm
            clipping (bool): Whether to clip the particle positions and velocities """
        # Get the model parameters and store them as a list in optimizer
        self.params = [param for param in model.parameters()]
        # Initialize the optimizer with the parameters and defaults
        defaults = dict(weight_range=weight_range, num_particles=num_particles, clipping=False)
        super(ParticleSwarm, self).__init__(self.params, defaults)

        # Store the model
        self.model = model
        # Store the parameters
        self.weight_range = weight_range
        self.num_particles = num_particles
        # Initialize cognitive coef, social coef, and inertia
        self.c1 = None
        self.c2 = None
        self.w = None
        # Initialize clipping, bool to clip the particle positions and velocities
        self.clip = clipping
        # Velocity max, if clipping is enabled
        self.v_max = 0.1
        # Maximum and minimum values for the particle positions, if clipping is enabled
        self.p_min = -weight_range / 2
        self.p_max = weight_range / 2
        # Diversification factor used to reinitialize particles to maintain diversity
        self.diversity_factor = 0.1
        # Initialize the particles
        self.particles = self.generate_particles(self.params, self.num_particles, self.weight_range)
        # Initialize the global best position and fitness
        self.global_best_position = self._get_model_parameters()
        self.global_best_fitness = float('inf')
        # Move particle dictionary to either move and clip or move no clip of the particle
        self.move_particle = {
            # If clipping is enabled, move and clip the particle
            True: self.clip_move,
            # If clipping is disabled, just move the particle
            False: self.move
        }

    # Get the model parameters
    def _get_model_parameters(self):
        """ Get the model parameters
        Returns:
            List: List of numpy arrays containing the model parameters """
        # Return the model parameters as a list of numpy arrays
        return [param.data.clone().detach().cpu().numpy() for param in self.model.parameters()]

    # Set the model parameters
    def _set_model_parameters(self, individual):
        """ Set the model parameters
        Args:
            individual (List): List of numpy arrays containing the model parameters """
        # Set the model parameters from a list of numpy arrays
        for param, individual_param in zip(self.model.parameters(), individual):
            param.data = torch.tensor(individual_param, dtype=param.data.dtype, device=param.data.device)

    # Generate particles
    def generate_particles(self, params, num_particles, weight_range):
        """ Generate particles for the particle swarm
        Args:
            params (List): List of model parameters
            num_particles (int): Number of particles in the swarm
            weight_range (int): Range of the initial weights for the particles
        Returns:
            particles (List): List of particles in the swarm """
        # Initialize the particles
        particles = []
        # Loop through the number of particles
        for i in range(num_particles):
            # Generate a new particle within the weight range using initial weights
            # as reference for the shape of the particle
            new_particle = Particle(params, weight_range)
            # Append the new particle to the particles
            particles.append(new_particle)
        # Return the particles
        return particles

    # Objective function
    def objective_function(self, outputs, y):
        """ Objective function for the particle swarm
        Args:
            X (Tensor): Input data
            y (Tensor): Target data
        Returns:
            Loss (float): Loss for the model """
        # Calculate the loss
        loss = self.model.criterion(outputs, y)
        '''# No gradient computation is needed
        with torch.no_grad():
            # Feed the input data to the model
            outputs = self.model(X)'''

        # Return the loss
        return loss.item()

    # Clip and move particle
    def clip_move(self, particle, cognitive_velocity, social_velocity):
        """ Clip and move particle
        Args:
            particle (Particle): Particle to be moved
            cognitive_velocity (List): Cognitive velocity for the particle
            social_velocity (List): Social velocity for the particle """
        # Update velocity and clip within the velocity max
        particle.velocity = [np.clip(w * v + cv + sv, -self.v_max, self.v_max) for w, v, cv, sv in
                             zip(self.w, particle.velocity, cognitive_velocity, social_velocity)]
        # Update and clip position within the particle min and max
        particle.position = [np.clip(p + v, self.p_min, self.p_max) for p, v in
                             zip(particle.position, particle.velocity)]

    # Move particle no clipping
    def move(self, particle, cognitive_velocity, social_velocity):
        """ Move particle without clipping
        Args:
            particle (Particle): Particle to be moved
            cognitive_velocity (List): Cognitive velocity for the particle
            social_velocity (List): Social velocity for the particle """
        # Update velocity
        particle.velocity = [w * v + cv + sv for w, v, cv, sv in
                             zip(self.w, particle.velocity, cognitive_velocity, social_velocity)]
        # Update position
        particle.position = [p + v for p, v in zip(particle.position, particle.velocity)]

    # Step function
    def step(self, closure=None):
        """ Step function for the particle swarm
        Args:
            closure (callable): A closure that reevaluates the model and returns the loss
        Returns:
            Best solution fitness score (float) """
        # If closure is not provided, raise an error
        if closure is None:
            raise ValueError("Closure is required for PSO step")



        # Loop through the particles
        for particle in self.particles:

            # Get random numbers to
            r1, r2 = np.random.rand(2)
            # Calculate cognitive and social velocities
            cognitive_velocity = [self.c1 * r1 * (bp - p) for bp, p in zip(particle.best_position, particle.position)]
            social_velocity = [self.c2 * r2 * (gbp - p) for gbp, p in zip(self.global_best_position, particle.position)]

            # Update position, clip if specified
            self.move_particle[self.clip](particle, cognitive_velocity, social_velocity)
            # Set the model parameters to the particle position
            self._set_model_parameters(particle.position)
            # Get the input data and target data from the closure
            X, y, outputs = closure()
            # Evaluate fitness
            current_fitness = self.objective_function(outputs, y)
            # Update personal best
            if current_fitness < particle.best_fitness:
                particle.best_fitness = current_fitness
                particle.best_position = particle.position.copy()
            # Update global best
            if current_fitness < self.global_best_fitness:
                self.global_best_fitness = current_fitness
                self.global_best_position = particle.position.copy()

            '''# Reinitialize particles for diversity if stagnation is detected
            if np.random.rand() < self.diversity_factor:
                particle.position = [np.random.uniform(self.p_min, self.p_max, size=p.shape).astype(np.float32) for
                                     p in particle.position]
                particle.velocity = [np.random.uniform(-self.v_max, self.v_max, size=v.shape).astype(np.float32) for
                                     v in particle.velocity]'''
        # Return the global best fitness
        return self.global_best_fitness


# Particle class for Particle Swarm Optimization
class Particle:
    """ Particle class for Particle Swarm Optimization
    Args:
        params (List): List of model parameters for reference in intitializing the particle position and velocity
        weight_range (int): Range of the initial weights for the particles """
    def __init__(self, params, weight_range):
        # Initialize the particle with random position and velocity within the weight range
        low = -weight_range / 2
        high = weight_range / 2
        # Initialize the position and velocity withing the weight range
        self.position = [np.random.uniform(low, high, size=param.size()).astype(np.float32) for param in params]
        self.velocity = [np.random.uniform(low, high, size=param.size()).astype(np.float32) for param in params]
        # Initialize the personal best position and fitness
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

