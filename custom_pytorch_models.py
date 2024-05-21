import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import custom_pytorch_optimizers as cust_optims


class XOR_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(XOR_Model, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.train_losses = []
        self.train_accuracies = []

        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.sigmoid(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

    @staticmethod
    def calculate_accuracy(predictions, targets):
        rounded_preds = torch.round(predictions)
        correct = (rounded_preds == targets).float()
        accuracy = correct.sum() / len(correct)
        return accuracy.item() * 100

    def train_model(self, X, y, epochs=10000, lr=0.1):
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            optimizer.step()

            accuracy = self.calculate_accuracy(outputs, y)
            self.train_losses.append(loss.item())
            self.train_accuracies.append(accuracy)

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}, Accuracy: {accuracy:.2f}%')

    def plot_metrics(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(12, 5))

        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss vs. Epochs')
        plt.legend()

        # Plot training accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label='Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy vs. Epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()


class XOR_GA_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(XOR_GA_Model, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.train_losses = []
        self.train_accuracies = []

        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.sigmoid(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

    @staticmethod
    def calculate_accuracy(predictions, targets):
        rounded_preds = torch.round(predictions)
        correct = (rounded_preds == targets).float()
        accuracy = correct.sum() / len(correct)
        return accuracy.item() * 100

    def train_model(self, X, y, epochs=10000, population_size=20, mutation_rate=0.3, weight_range=40):
        optimizer = cust_optims.GeneticAlgorithmOptimizer(self, population_size, mutation_rate, weight_range)

        for epoch in range(epochs):
            def closure():
                optimizer.zero_grad()
                outputs = self(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                accuracy = self.calculate_accuracy(outputs, y)
                self.train_losses.append(loss.item())
                self.train_accuracies.append(accuracy)
                return X, y

            best_fitness = optimizer.step(closure)
            print(f"Generation {epoch + 1}, Best Fitness: {best_fitness}")


            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {self.train_losses[-1]:.6f}, Accuracy: {self.train_accuracies[-1]:.2f}%')

    def plot_metrics(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(12, 5))

        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss vs. Epochs')
        plt.legend()

        # Plot training accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label='Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy vs. Epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()