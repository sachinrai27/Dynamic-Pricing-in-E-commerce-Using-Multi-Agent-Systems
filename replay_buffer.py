import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        # Add the experience (state, action, reward, next_state) to the buffer.
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # Randomly sample a batch of experiences from the buffer.
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        # Return the current size of the buffer.
        return len(self.buffer)
