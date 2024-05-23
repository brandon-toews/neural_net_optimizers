import numpy as np
import copy
from numba import njit


class GeneticAlgorithm:
    def __init__(self, neural_net, population_size=20, mutation_rate=0.3, weight_range=40):
        """
        Initialize the genetic algorithm optimizer.

        Parameters:
        neural_net: Neural network to optimize.
        population_size (int): Number of individuals in the population.
        mutation_rate (float): Probability of mutation.
        weight_range (int): Range of initial weights.
        """
        self.nn = neural_net
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.weight_range = weight_range
        self.population = self.generate_population(self.nn.get_parameters(), population_size, self.nn.data_type)
        self.fitness_scores = [0] * population_size
        self.criterion = self.nn.criterion
        self.loss_function = {
            'MSE': self.MSE,
            'CrossEntropy': self.CrossEntropy
        }

    @staticmethod
    def generate_population(initial_individual, population_size, data_type, weight_range=2):
        """
        Generate an initial population with random weights.

        Parameters:
        initial_individual (np.ndarray): Initial set of parameters.
        population_size (int): Number of individuals in the population.
        data_type (type): Data type for the parameters.
        weight_range (int): Range for random weights.

        Returns:
        list: Generated population.
        """
        low = int(np.iinfo(data_type).min * 0.66)
        high = int(np.iinfo(data_type).max * 0.66)
        population = []
        for _ in range(population_size):
            # Create a new individual by adding random noise to the initial individual
            new_individual = initial_individual + np.random.randint(low, high, initial_individual.shape).astype(
                data_type)
            population.append(new_individual)
        return population

    @staticmethod
    def neuron_crossover(parent1, parent2, weight_shapes, bias_shapes):
        """
        Perform neuron-level crossover between two parents to produce two children.

        Parameters:
        parent1 (np.ndarray): Parameters of the first parent.
        parent2 (np.ndarray): Parameters of the second parent.
        weight_shapes (list): List of weight shapes.
        bias_shapes (list): List of bias shapes.

        Returns:
        tuple: Two children resulting from crossover.
        """
        child1 = np.copy(parent1)
        child2 = np.copy(parent2)

        index = 0
        for w_shape, b_shape in zip(weight_shapes, bias_shapes):
            num_neurons = w_shape[1]
            w_size = np.prod(w_shape)
            b_size = np.prod(b_shape)

            # Swap weights and biases for each neuron
            for i in range(num_neurons):
                if np.random.rand() < 0.5:
                    # Swap weights
                    start = index + i * w_shape[0]
                    end = start + w_shape[0]
                    child1[start:end] = parent2[start:end]
                    child2[start:end] = parent1[start:end]

                    # Swap biases
                    start = index + w_size + i
                    child1[start] = parent2[start]
                    child2[start] = parent1[start]

            index += w_size + b_size

        return child1, child2

    @staticmethod
    def layer_crossover(parent1, parent2, weight_shapes, bias_shapes):
        """
        Perform layer-level crossover between two parents to produce two children.

        Parameters:
        parent1 (np.ndarray): Parameters of the first parent.
        parent2 (np.ndarray): Parameters of the second parent.
        weight_shapes (list): List of weight shapes.
        bias_shapes (list): List of bias shapes.

        Returns:
        tuple: Two children resulting from crossover.
        """
        child1 = np.copy(parent1)
        child2 = np.copy(parent2)

        index = 0
        for w_shape, b_shape in zip(weight_shapes, bias_shapes):
            w_size = np.prod(w_shape)
            b_size = np.prod(b_shape)

            # Crossover for weights
            mask = np.random.rand(w_size) > 0.5
            child1[index:index + w_size][mask] = parent2[index:index + w_size][mask]
            child2[index:index + w_size][mask] = parent1[index:index + w_size][mask]
            index += w_size

            # Crossover for biases
            mask = np.random.rand(b_size) > 0.5
            child1[index:index + b_size][mask] = parent2[index:index + b_size][mask]
            child2[index:index + b_size][mask] = parent1[index:index + b_size][mask]
            index += b_size

        return child1, child2

    @staticmethod
    def crossover(parent1, parent2):
        """
        Perform element-wise crossover between two parents to produce two children.

        Parameters:
        parent1 (np.ndarray): Parameters of the first parent.
        parent2 (np.ndarray): Parameters of the second parent.

        Returns:
        tuple: Two children resulting from crossover.
        """
        crossover_mask = np.random.rand(parent1.shape[0]) < 0.5
        child1, child2 = np.copy(parent1), np.copy(parent2)
        child1[crossover_mask] = parent2[crossover_mask]
        child2[crossover_mask] = parent1[crossover_mask]
        return child1, child2

    @staticmethod
    def mutate(individual, mutation_rate, data_type):
        """
        Mutate an individual by altering its parameters with a given probability.

        Parameters:
        individual (np.ndarray): Individual to mutate.
        mutation_rate (float): Probability of mutation.
        data_type (type): Data type for the parameters.

        Returns:
        np.ndarray: Mutated individual.
        """
        if np.random.rand() < mutation_rate:
            min_delta = np.iinfo(data_type).min
            max_delta = np.iinfo(data_type).max

            half_size = len(individual) // 2
            weights = individual[:half_size]
            biases = individual[half_size:]

            # Mutate weights
            delta_weights = np.random.randint(min_delta, max_delta, weights.shape).astype(data_type)
            weights += delta_weights

            # Mutate biases
            delta_biases = np.random.randint(-1, 2, biases.shape).astype(data_type)
            biases += delta_biases
            np.clip(biases, -5, 5, out=biases)

            # Combine weights and biases back into a single individual
            mutated_individual = np.concatenate((weights, biases))

            return mutated_individual
        else:
            return individual

    def fitness_function(self, X, y):
        """
        Compute the fitness of the neural network.

        Parameters:
        X (np.ndarray): Input data.
        y (np.ndarray): True labels.

        Returns:
        float: Fitness score.
        """
        predictions = self.nn.forward(X)
        return self.loss_function[self.criterion](predictions, y)

    def MSE(self, predictions, y):
        """
        Compute Mean Squared Error.

        Parameters:
        predictions (np.ndarray): Predicted values.
        y (np.ndarray): True values.

        Returns:
        float: Computed MSE.
        """
        error = np.mean((predictions - y) ** 2)
        self.nn.train_losses.append(error)
        return 1 / (error + 1)

    @staticmethod
    def CrossEntropy(predictions, y):
        """
        Compute Cross Entropy loss.

        Parameters:
        predictions (np.ndarray): Predicted values.
        y (np.ndarray): True values.

        Returns:
        float: Computed Cross Entropy loss.
        """
        epsilon = 1e-7
        loss = -np.sum(y * np.log(predictions + epsilon)) / y.shape[0]
        return 1 / (loss + 1)

    def train(self, X, y, generations=50):
        """
        Train the neural network using the genetic algorithm.

        Parameters:
        X (np.ndarray): Training data.
        y (np.ndarray): Training labels.
        generations (int): Number of generations to train.

        Returns:
        list: Training progress over generations.
        """
        training_progress = []
        best_solution = (-np.inf, None)
        weight_shapes = [w.shape for w in self.nn.weights]
        bias_shapes = [b.shape for b in self.nn.biases]

        for generation in range(generations):
            fitness_scores = []

            # Evaluate fitness of each individual
            for individual in self.population:
                self.nn.set_parameters(individual)
                score = self.fitness_function(X, y)
                if score > best_solution[0]:
                    best_solution = (score, copy.deepcopy(individual))
                fitness_scores.append(score)

            # Sort the population by fitness scores in descending order
            sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
            sorted_population = [self.population[i] for i in sorted_indices]

            # Select the top 50% of the population
            self.population = sorted_population[:self.population_size // 2]

            new_population = []

            # Perform crossover and mutation to create new individuals
            while len(new_population) < self.population_size:
                parent1, parent2 = (self.population[np.random.randint(len(self.population))],
                                    self.population[np.random.randint(len(self.population))])
                child1, child2 = self.neuron_crossover(parent1, parent2, weight_shapes, bias_shapes)
                new_population.append(self.mutate(child1, self.mutation_rate, self.nn.data_type))
                new_population.append(self.mutate(child2, self.mutation_rate, self.nn.data_type))

            self.population = new_population
            best_fitness = max(fitness_scores)
            training_progress.append(best_fitness)
            print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

        # Set the parameters of the neural network to the best solution found
        self.nn.set_parameters(best_solution[1])
        predictions = self.nn.forward(X)
        np.set_printoptions(precision=4, suppress=True)
        print("Predictions after genetic optimization:")
        print(predictions)
        print(f'Accuracy: {best_solution[0]}')

        return training_progress


class ParticleSwarm:
    def __init__(self, neural_net, weight_range=1300, num_particles=20, c1=2.0, c2=2.0, w=0.5, v_max=0.1):
        """
        Initialize the particle swarm optimizer.

        Parameters:
        neural_net: Neural network to optimize.
        weight_range (int): Range for initial particle weights.
        num_particles (int): Number of particles in the swarm.
        c1 (float): Cognitive coefficient.
        c2 (float): Social coefficient.
        w (float): Inertia coefficient.
        v_max (float): Maximum velocity.
        """
        self.nn = neural_net
        self.weight_range = weight_range
        self.num_particles = num_particles
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.v_max = v_max
        self.p_min = np.iinfo(self.nn.data_type).min * 0.66
        self.p_max = np.iinfo(self.nn.data_type).max * 0.66
        self.particles = self.generate_particles(self.num_particles, self.weight_range)
        self.global_best_position = None
        self.global_best_fitness = float('inf')

    def generate_particles(self, num_particles, weight_range):
        """
        Generate an initial swarm of particles.

        Parameters:
        num_particles (int): Number of particles in the swarm.
        weight_range (int): Range for initial particle weights.

        Returns:
        list: Generated particles.
        """
        particles = []
        for i in range(num_particles):
            new_particle = Particle(self.nn.get_parameters(), weight_range, self.p_min, self.p_max, self.v_max,
                                    self.nn.data_type)
            particles.append(new_particle)
        return particles

    def fitness_function(self, X, y):
        """
        Compute the fitness of the neural network.

        Parameters:
        X (np.ndarray): Input data.
        y (np.ndarray): True labels.

        Returns:
        float: Fitness score.
        """
        predictions = self.nn.forward(X)
        return self.nn.calculate_loss(predictions, y)

    def step(self, X, y):
        """
        Perform one iteration of the particle swarm optimization.

        Parameters:
        X (np.ndarray): Input data.
        y (np.ndarray): True labels.

        Returns:
        float: Global best fitness.
        """
        if self.global_best_position is None:
            self.global_best_position = self.nn.get_parameters()

        for particle in self.particles:
            # Update particle velocity
            r1, r2 = np.random.rand(2)
            cognitive_velocity = self.c1 * r1 * (particle.best_position - particle.position)
            social_velocity = self.c2 * r2 * (self.global_best_position - particle.position)
            particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity

            # Clip velocity to maximum velocity
            particle.velocity = np.clip(particle.velocity, -self.v_max, self.v_max)

            # Update particle position
            particle.position += particle.velocity

            # Clip position to the allowed range
            particle.position = np.clip(particle.position, self.p_min, self.p_max)

            # Evaluate fitness
            self.nn.set_parameters(particle.position)
            fitness = self.fitness_function(X, y)

            # Update personal best
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = np.copy(particle.position)

            # Update global best
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = np.copy(particle.position)

        return self.global_best_fitness


class Particle:
    def __init__(self, initial_position, weight_range, p_min, p_max, v_max, data_type):
        """
        Initialize a particle.

        Parameters:
        initial_position (np.ndarray): Initial position of the particle.
        weight_range (int): Range for particle weights.
        p_min (float): Minimum position value.
        p_max (float): Maximum position value.
        v_max (float): Maximum velocity.
        data_type (type): Data type for the parameters.
        """
        self.position = np.random.uniform(p_min, p_max, initial_position.shape).astype(data_type)
        self.velocity = np.random.uniform(-v_max, v_max, initial_position.shape).astype(data_type)
        self.best_position = np.copy(self.position)
        self.best_fitness = float('inf')
