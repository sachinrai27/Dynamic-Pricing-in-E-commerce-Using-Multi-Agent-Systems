from dqnn import DQN
from replay_buffer import ReplayBuffer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class VendorAgent:
    def __init__(self, products, state_space, action_space, base_demand, epsilon=0.1, learning_rate=0.001, discount_factor=0.95, buffer_size=10000, batch_size=64):
        """
        Initializes the VendorAgent class.

        Args:
            products (list): List of Product objects representing the products available.
            state_space (int): Dimensionality of the state space.
            action_space (list): List of available actions.
            base_demand (float): Base demand value for the products.
            epsilon (float, optional): Exploration rate for action selection. Defaults to 0.1.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            discount_factor (float, optional): Discount factor for future rewards. Defaults to 0.95.
            buffer_size (int, optional): Size of the replay buffer. Defaults to 10000.
            batch_size (int, optional): Batch size for training. Defaults to 64.
        """
        self.products = products
        self.state_space = state_space
        self.action_space = action_space
        self.base_demand = base_demand
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_space, len(action_space)).to(self.device)
        self.target_network = DQN(state_space, len(action_space)).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.total_rewards = 0
        self.history = {'actions': [], 'rewards': []}

    def choose_action(self, state):
        """
        Chooses an action based on the current state using an epsilon-greedy policy.

        Args:
            state (list): Current state representation.

        Returns:
            int: Chosen action.
        """
        self.epsilon = max(0.01, self.epsilon * 0.999)
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state_tensor = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor.unsqueeze(0))
            action = self.action_space[q_values.argmax().item()]
        return action

    def adjust_prices(self, state):
        """
        Adjusts the prices of the products based on the chosen action.

        Args:
            state (list): Current state representation.

        Returns:
            int: Chosen action.
        """
        action = self.choose_action(state)
        for product in self.products:
            potential_price = product.current_price * (1 + action / 100)
            if potential_price >= product.cogs * 1.1:
                product.current_price = potential_price
        self.history['actions'].append(action)  # update history
        return action

    def update_q_network(self):
        """
        Performs a Q-network update using a batch of experiences from the replay buffer.
        """
        if len(self.buffer) < self.batch_size:
            return
        samples = self.buffer.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state = zip(*samples)
        state_tensor = torch.FloatTensor(batch_state).to(self.device)
        next_state_tensor = torch.FloatTensor(batch_next_state).to(self.device)
        action_indices = [self.action_space.index(a) for a in batch_action]
        action_tensor = torch.LongTensor(action_indices).unsqueeze(1).to(self.device)
        reward_tensor = torch.FloatTensor(batch_reward).to(self.device)

        q_values = self.q_network(state_tensor)
        q_values = q_values.gather(1, action_tensor)

        with torch.no_grad():
            next_q_values = self.target_network(next_state_tensor).max(1)[0]

        expected_q_values = reward_tensor + self.discount_factor * next_q_values

        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn(self, state, action, reward, next_state):
        """
        Learns from an experience by updating the replay buffer and Q-network.

        Args:
            state (list): Current state representation.
            action (int): Chosen action.
            reward (float): Obtained reward.
            next_state (list): Next state representation.
        """
        self.buffer.push(state, action, reward, next_state)
        self.history['rewards'].append(reward)  # update history
        self.update_q_network()