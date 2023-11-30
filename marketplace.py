import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Marketplace:
    def __init__(self, vendors):
        """
        Initializes the Marketplace class.

        Args:
            vendors (list): List of VendorAgent objects representing the vendors in the marketplace.
        """
        self.vendors = vendors
        self.history = {'average_profit': []}

    def simulate(self, episodes, time_steps):
        """
        Simulates the marketplace for a specified number of episodes and time steps.

        Args:
            episodes (int): Number of episodes to simulate.
            time_steps (int): Number of time steps per episode.
        """
        self.total_rewards = 0
        for episode in range(episodes):
            total_profit = 0
            for vendor in self.vendors:
                state = self.get_current_state(vendor)
                for _ in range(time_steps):
                    season = self.get_season(episode)
                    action = vendor.adjust_prices(state)
                    next_state, reward = self.step(vendor, action, season)
                    vendor.learn(state, action, reward, next_state)
                    state = next_state
                    self.total_rewards += reward
                    total_profit += reward
                self.update_target_network()
            average_profit = total_profit / len(self.vendors)  # calculate average profit
            self.history['average_profit'].append(average_profit)

    def get_season(self, episode):
        """
        Determines the season based on the episode number.

        Args:
            episode (int): Current episode number.

        Returns:
            str: Season name.
        """
        day_of_year = episode % 360
        if day_of_year < 90 or day_of_year >= 270:
            return 'winter'
        elif day_of_year < 180:
            return 'spring/fall'
        else:
            return 'summer'

    def get_current_state(self, vendor_to_ignore):
        """
        Gets the current state representation of the marketplace.

        Args:
            vendor_to_ignore (VendorAgent): VendorAgent object to exclude from the state calculation.

        Returns:
            list: Current state representation.
        """
        prices = [product.current_price for vendor in self.vendors for product in vendor.products if vendor != vendor_to_ignore]
        total_price = sum(prices)
        min_price = min(prices)
        max_price = max(prices)
        average_price = total_price // len(prices) if prices else 0  # avoid division by zero
        return [int(min(self.vendors[0].state_space - 1, average_price)), min_price, max_price]


    def step(self, vendor, action, season):
        """
        Performs a time step in the marketplace simulation.

        Args:
            vendor (VendorAgent): VendorAgent object performing the action.
            action (int): Chosen action.
            season (str): Current season.

        Returns:
            tuple: Next state representation and reward obtained.
        """
        next_state = self.get_current_state(vendor)
        average_price = sum(product.current_price for other_vendor in self.vendors for product in
                        other_vendor.products if other_vendor != vendor) / (len(self.vendors) - 1)
        reward = 0
        for product in vendor.products:
            if season == 'winter':
                demand_factor = 1.5
            elif season == 'summer':
                demand_factor = 0.5
            else:
                demand_factor = 1

            if product.current_price < average_price:
                demand_factor *= (average_price / product.current_price) ** 0.5
            else:
                demand_factor *= (average_price / product.current_price) ** 2

            quantity_sold = product.base_demand * demand_factor  # use product's base demand instead of vendor's
            profit = quantity_sold * (product.current_price - product.cogs)
            reward += profit / 1000  # scaling the reward

            if product.current_price < product.cogs * 1.1:
                reward -= product.current_price * 0.001  # scaling the penalty
            
            # Penalty for too high prices
            if product.current_price > product.cogs * 2:
                reward -= product.current_price * 0.001  # scaling the penalty
        
            # Penalty for extreme price changes
            if abs(product.current_price - product.cogs) > product.cogs:
                reward -= 0.5

        return next_state, reward

    def update_target_network(self):
        """
        Updates the target network of each vendor in the marketplace.
        """
        for vendor in self.vendors:
            vendor.target_network.load_state_dict(vendor.q_network.state_dict())