import numpy as np
import copy


class GeneticAlgorithm:
    def __init__(self, neural_net, population_size=20, mutation_rate=0.3, weight_range=40):
        self.nn = neural_net
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.weight_range = weight_range
        self.population = self.generate_population(self.nn.get_parameters(), population_size, weight_range)
        self.fitness_scores = [0] * population_size

    # Function to generate the initial population
    def generate_population(self, initial_individual, population_size, weight_range=2):
        low = np.int8(-weight_range / 2)
        high = np.int8(weight_range / 2)
        population = []
        for i in range(population_size):
            new_individual = ([], [])
            for weights, biases in zip(initial_individual[0], initial_individual[1]):
                layer_weight_shape = weights.shape
                layer_biases_shape = biases.shape
                new_individual[0].append(
                    np.random.randint(low, high, (layer_weight_shape[0], layer_weight_shape[1]), np.int8))
                new_individual[1].append(np.zeros(layer_biases_shape, np.int8))

            population.append(new_individual)

        return population

    def fitness_function(self, X, y):
        predictions = self.nn.forward(X)
        # criterion = tnn.MSELoss()
        # loss = criterion(predictions, y)
        error = np.mean((predictions - y) ** 2)
        return 1 / (error + 1)

    def crossover(self, parent1, parent2):
        child1, child2 = [[], []], [[], []]
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

        return child1, child2

    def mutate(self, individual, max_delta=10):
        """
        Mutates an individual's weights and biases by slightly adjusting them.

        Args:
            individual: The neural network weights and biases to mutate.
            max_delta: The maximum change that can be applied to a weight.

        Returns:
            The mutated individual.
        """
        for which_layer in range(len(individual[0])):
            if np.random.rand() < self.mutation_rate:
                # low = -weight_range/2

                which_neuron = np.random.randint(individual[0][which_layer].shape[0])
                for which_weight in range(individual[0][which_layer].shape[1]):
                    # Generate a small change, ensuring it's within the range [-max_delta, max_delta]
                    delta = np.random.randint(-max_delta, max_delta, 1, np.int8)
                    # Apply the change to the weight
                    individual[0][which_layer][which_neuron, which_weight] += delta[0]
                    # Optional: Clip the weights to ensure they remain within a desired range, e.g., [-1, 1]
                    # individual[which_layer][which_neuron, which_weight] = np.clip(
                    # individual[which_layer][which_neuron, which_weight], -20, 20)

                which_bias = np.random.randint(individual[1][which_layer].shape)
                # Generate a small change, ensuring it's within the range [-max_delta, max_delta]
                delta = np.random.randint(-1, 2, 1, np.int8)
                individual[1][which_layer][which_bias] += delta
                individual[1][which_layer][which_bias] = np.clip(
                    individual[1][which_layer][which_bias], -1, 1
                )
        return individual

    # Function to train the neural network using genetic algorithm
    def train(self, X, y, generations=50):
        training_progress = []
        # Initialize the best solution found so far
        best_solution = (-np.inf, None)
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
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1))
                new_population.append(self.mutate(child2))

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
