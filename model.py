# model.py - Definition of the MLP neural network architecture for MNIST classification.

import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) neural network for image classification.

    A feedforward neural network with three hidden layers designed to process
    flattened image inputs (784 features) and output class predictions (10 classes).

    Architecture:
        - Input layer: 784 features (28x28 image flattened)
        - Hidden layer 1: 512 units with ReLU activation
        - Hidden layer 2: 128 units with ReLU activation
        - Hidden layer 3: 32 units with ReLU activation
        - Output layer: 10 units (logits for 10 classes)

    Attributes:
        fc1 (nn.Linear): First fully connected layer (784 -> 512)
        fc2 (nn.Linear): Second fully connected layer (512 -> 128)
        fc3 (nn.Linear): Third fully connected layer (128 -> 32)
        out (nn.Linear): Output layer (32 -> 10)

    Methods:
        forward(x): Processes input tensor through the network
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28) or (batch_size, 784)
            Returns:
                torch.Tensor: Output logits of shape (batch_size, 10)
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.out = nn.Linear(32, 10)

    def forward(self, x):
        """
        Forward pass through the neural network.
        Args:
            x: Input tensor of shape (batch_size, 784) or any shape that can be reshaped to (batch_size, 784).
        Returns:
            Output tensor from the final output layer.
        """

        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)
