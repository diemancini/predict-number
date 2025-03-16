import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


class MacrosNN:
    PREDICTED_DIGIT_KEY = "predicted_digit"
    CONFIDENCE_KEY = "confidence"
    _BATCH_SIZE = 64
    _LEARNING_RATE = 0.001
    _NUM_EPOCHS = 5
    _MEAN = (0.5,)
    _STD = (0.5,)
    _MNIST_MODEL_TRAINED_PATH = "mnist_model.pth"
    _DATASET_PATH = "./data"


class TrainingNumbersNN(MacrosNN):

    def _setup_training_number(self):

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    self._MEAN, self._STD
                ),  # Normalize with mean and std of MNIST
            ]
        )

        # Load MNIST dataset
        train_dataset = MNIST(
            root=self._DATASET_PATH, train=True, transform=transform, download=True
        )
        test_dataset = MNIST(
            root=self._DATASET_PATH, train=False, transform=transform, download=True
        )

        # loader_nn = {}

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self._BATCH_SIZE, shuffle=True
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=self._BATCH_SIZE, shuffle=False
        )

        return {"train": train_loader, "test": test_loader}

    def train_numbers(self):

        loader = self._setup_training_number()

        # Initialize the model, loss function, and optimizer
        model = SimpleNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self._LEARNING_RATE)

        # Training loop
        for epoch in range(self._NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            for i, (images, labels) in enumerate(loader["train"]):
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if (i + 1) % 100 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{self._NUM_EPOCHS}], Step [{i + 1}/{len(loader['train'])}], Loss: {loss.item():.4f}"
                    )

            print(
                f"Epoch [{epoch + 1}/{self._NUM_EPOCHS}], Average Loss: {running_loss / len(loader['train']):.4f}"
            )

        try:
            torch.save(model.state_dict(), self._MNIST_MODEL_TRAINED_PATH)
            shutil.rmtree(self._DATASET_PATH)
        except RuntimeError as e:
            print(e)

    def evaluation(self, model, test_loader):

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            f"Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%"
        )


# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":

    print("Starting train the number model...")
    tn = TrainingNumbersNN()
    tn.train_numbers()
