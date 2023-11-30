# Defining the neural network architecture for Deep Q-Network (DQN).
import torch
import torch.nn as nn
import random

class DQN(nn.Module):
    def __init__(self, state_space, action_space):
        # Initializing the parent class (nn.Module).
        super(DQN, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.fc1 = nn.Linear(state_space, 64)  # First fully connected layer with 64 output units.
        self.fc2 = nn.Linear(64, 128)  # Second fully connected layer with 128 output units.
        self.fc3 = nn.Linear(128, action_space)  # Third fully connected layer with action_space number of output units.

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to the output of the first layer.
        x = torch.relu(self.fc2(x))  # Apply ReLU activation to the output of the second layer.
        x = self.fc3(x)  # Output the final Q-values without any activation function.
        return x