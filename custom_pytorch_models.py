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
                # optimizer.zero_grad()
                outputs = self(X)
                loss = self.criterion(outputs, y)
                # loss.backward()
                accuracy = self.calculate_accuracy(outputs, y)
                self.train_losses.append(loss.item())
                self.train_accuracies.append(accuracy)
                return X, y, outputs

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

class Mnist_Model(nn.Module):
    def __init__(self, lr=0.001):
        super(Mnist_Model, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)
        self.train_losses = []
        self.train_accuracies = []

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


    @staticmethod
    def calculate_accuracy(predictions, targets):
        #self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            _, predicted = torch.max(predictions.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        # print(f'Accuracy: {accuracy}%')
        return accuracy

    def evaluate_model(self, test_loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy}%')
        return accuracy

    def train_model(self, train_loader, epochs=5):
        self.train()
        print('Training the model ...')
        for epoch in range(epochs):
            running_loss = 0.0
            running_accuracy = 0.0
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                accuracy = self.calculate_accuracy(outputs, labels)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                running_accuracy += accuracy

            self.train_losses.append(running_loss / len(train_loader))
            self.train_accuracies.append(running_accuracy / len(train_loader))
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Accuracy: {running_accuracy / len(train_loader):.2f}%")




            # if (epoch + 1) % 100 == 0:
                # print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}, Accuracy: {accuracy:.2f}%')

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


class Mnist_GA_Model(nn.Module):
    def __init__(self, population_size=20, mutation_rate=0.3, weight_range=40):
        super(Mnist_GA_Model, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)
        self.train_losses = []
        self.train_accuracies = []

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = cust_optims.GeneticAlgorithmOptimizer(self, population_size, mutation_rate, weight_range)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


    @staticmethod
    def calculate_accuracy(predictions, targets):
        #self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            _, predicted = torch.max(predictions.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        # print(f'Accuracy: {accuracy}%')
        return accuracy

    def evaluate_model(self, test_loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy}%')
        return accuracy

    def train_model(self, train_loader, epochs=5):
        self.train()
        print('Training the model ...')

        for epoch in range(epochs):
            running_loss = 0.0
            running_accuracy = 0.0
            total_samples = 0
            for inputs, labels in train_loader:
                def closure():
                    #self.optimizer.zero_grad()
                    with torch.no_grad():
                        outputs = self(inputs)
                    #self.train_losses.append(loss.item())
                    #self.train_accuracies.append(accuracy)
                    return inputs, labels, outputs

                _, _, outputs = closure()
                # print(f"Outputs shape: {outputs.shape}")  # Expected: [batch_size, num_classes]
                # print(f"Labels shape: {labels.shape}")  # Expected: [batch_size]

                loss, accuracy = self.criterion(outputs, labels), self.calculate_accuracy(outputs, labels)
                running_loss += loss.item()
                running_accuracy += accuracy
                total_samples += 1

                best_fitness = self.optimizer.step(closure)
            avg_loss = running_loss / total_samples
            avg_accuracy = running_accuracy / total_samples
            self.train_losses.append(avg_loss/len(train_loader))
            self.train_accuracies.append(avg_accuracy/len(train_loader))
            print(f"Generation {epoch + 1}, Best Fitness: {best_fitness:.4f}")
            # self.train_losses.append(running_loss / len(train_loader))
            # self.train_accuracies.append(running_accuracy / len(train_loader))
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Accuracy: {running_accuracy / len(train_loader):.2f}%")

            # if (epoch + 1) % 100 == 0:
                # print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}, Accuracy: {accuracy:.2f}%')

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
