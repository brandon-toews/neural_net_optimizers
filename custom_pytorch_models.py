import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class XOR_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(XOR_Model, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.train_losses = []
        self.train_accuracies = []

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
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X)
            loss = criterion(outputs, y)
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
