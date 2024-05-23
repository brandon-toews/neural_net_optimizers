import numpy as np
import copy
from numba import njit

class GeneticAlgorithm:
    def __init__(self, neural_net, population_size=20, mutation_rate=0.3, weight_range=40):
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
        low = int(np.iinfo(data_type).min * 0.66) # -self.nn.data_type(weight_range / 2)
        high = int(np.iinfo(data_type).max * 0.66) # self.nn.data_type(weight_range / 2)
        population = []
        for _ in range(population_size):
            # Create a new individual by adding random noise to the initial individual
            new_individual = initial_individual + np.random.randint(low, high, initial_individual.shape).astype(data_type)
            population.append(new_individual)
        return population

    @staticmethod
    def neuron_crossover(parent1, parent2, weight_shapes, bias_shapes):
        child1 = np.copy(parent1)
        child2 = np.copy(parent2)

        index = 0
        for w_shape, b_shape in zip(weight_shapes, bias_shapes):
            num_neurons = w_shape[1]
            w_size = np.prod(w_shape)
            b_size = np.prod(b_shape)

            # For each neuron in the layer, swap weights and biases
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
        crossover_mask = np.random.rand(parent1.shape[0]) < 0.5
        child1, child2 = np.copy(parent1), np.copy(parent2)
        child1[crossover_mask] = parent2[crossover_mask]
        child2[crossover_mask] = parent1[crossover_mask]
        return child1, child2


    @staticmethod
    def mutate(individual, mutation_rate, data_type):
        if np.random.rand() < mutation_rate:
            min_delta = np.iinfo(data_type).min
            max_delta = np.iinfo(data_type).max

            half_size = len(individual) // 2
            weights = individual[:half_size]
            biases = individual[half_size:]

            # Mutate weights
            # mutation_mask_weights = np.random.rand(weights.shape[0]) < 0.5
            delta_weights = np.random.randint(min_delta, max_delta, weights.shape).astype(data_type)
            #weights[mutation_mask_weights] += delta_weights[mutation_mask_weights]
            weights += delta_weights


            # Mutate biases
            # mutation_mask_biases = np.random.rand(biases.shape[0]) < 0.5
            delta_biases = np.random.randint(-1, 2, biases.shape).astype(data_type)
            # biases[mutation_mask_biases] += delta_biases[mutation_mask_biases]
            biases += delta_biases
            np.clip(biases, -5, 5, out=biases)

            # Combine weights and biases back into a single individual
            mutated_individual = np.concatenate((weights, biases))

            return mutated_individual
        else:
            return individual

    def fitness_function(self, X, y):
        predictions = self.nn.forward(X)
        return self.loss_function[self.criterion](predictions, y)
        # criterion = tnn.MSELoss()
        # loss = criterion(predictions, y)


    def MSE(self, predictions, y):
        error = np.mean((predictions - y) ** 2)
        self.nn.train_losses.append(error)
        return 1 / (error + 1)

    @staticmethod
    def CrossEntropy(predictions, y):
        # loss = self.nn.calculate_loss(predictions, y)
        epsilon = 1e-7
        loss = -np.sum(y * np.log(predictions + epsilon)) / y.shape[0]
        #return loss
        return 1 / (loss + 1)



    # Function to train the neural network using genetic algorithm
    def train(self, X, y, generations=50):
        training_progress = []
        # Initialize the best solution found so far
        best_solution = (-np.inf, None)
        weight_shapes = [w.shape for w in self.nn.weights]
        bias_shapes = [b.shape for b in self.nn.biases]
        # Train the neural network
        for generation in range(generations):
            # Evaluate fitness of each individual
            fitness_scores = []
            # Evaluate fitness of each individual
            for individual in self.population:
                # Set the parameters of the neural network to the individual's weights and biases
                self.nn.set_parameters(individual)
                # Calculate the fitness score of the individual
                score = self.fitness_function(X, y)
                # Update the best solution found so far
                if score > best_solution[0]:
                    best_solution = (score, copy.deepcopy(individual))
                # Append the fitness score to the list of fitness scores
                fitness_scores.append(score)

            # Sort indices based on fitness_scores in descending order
            sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)

            # Reorder population based on sorted indices
            sorted_population = [self.population[i] for i in sorted_indices]
            # Select top 50% of the population
            self.population = sorted_population[:self.population_size // 2]

            # Create next generation
            new_population = []

            # Perform crossover and mutation to create new individuals
            while len(new_population) < self.population_size:
                # Select two parents randomly from the top population
                parent1, parent2 = (self.population[np.random.randint(len(self.population))],
                                    self.population[np.random.randint(len(self.population))])
                # Perform crossover to create two children
                child1, child2 = self.neuron_crossover(parent1, parent2, weight_shapes, bias_shapes)
                new_population.append(self.mutate(child1, self.mutation_rate, self.nn.data_type))
                new_population.append(self.mutate(child2, self.mutation_rate, self.nn.data_type))

            self.population = new_population
            best_fitness = max(fitness_scores)
            training_progress.append(best_fitness)
            print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

        # Test the best individual
        self.nn.set_parameters(best_solution[1])
        predictions = self.nn.forward(X)
        # Adjust numpy print options
        np.set_printoptions(precision=4, suppress=True)
        # Print the predictions
        print("Predictions after genetic optimization:")
        print(predictions)
        # Print the accuracy of the best solution
        print(f'Accuracy: {best_solution[0]}')

        return training_progress


class ParticleSwarm:
    def __init__(self, neural_net, weight_range=1300, num_particles=20, c1=2.0, c2=2.0, w=0.5, v_max=0.1):
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
        particles = []
        for i in range(num_particles):
            new_particle = Particle(self.nn.get_parameters(), weight_range, self.p_min, self.p_max, self.v_max, self.nn.data_type)
            particles.append(new_particle)
        return particles

    def fitness_function(self, X, y):
        predictions = self.nn.forward(X)
        return self.nn.calculate_loss(predictions, y)

    def step(self, X, y):
        if self.global_best_position is None:
            self.global_best_position = self.nn.get_parameters()

        for particle in self.particles:
            r1, r2 = np.random.rand(2)
            cognitive_velocity = self.c1 * r1 * (particle.best_position - particle.position)
            social_velocity = self.c2 * r2 * (self.global_best_position - particle.position)

            # Update and clip velocity
            particle.velocity = particle.velocity + cognitive_velocity + social_velocity

            # Update and clip position
            particle.position = particle.position + particle.velocity

            self.nn.set_parameters(particle.position)
            current_fitness = self.fitness_function(X, y)

            if current_fitness < particle.best_fitness:
                particle.best_fitness = current_fitness
                particle.best_position = copy.deepcopy(particle.position)

            if current_fitness < self.global_best_fitness:
                self.global_best_fitness = current_fitness
                self.global_best_position = copy.deepcopy(particle.position)

        self.nn.set_parameters(self.global_best_position)
        return self.global_best_fitness


class Particle:
    def __init__(self, initial_position, weight_range, p_min, p_max, v_max, data_type):
        self.position = np.random.uniform(p_min, p_max, initial_position.shape).astype(data_type)
        self.velocity = np.random.uniform(-v_max, v_max, initial_position.shape).astype(data_type)
        self.best_position = copy.deepcopy(self.position)
        self.best_fitness = float('inf')
