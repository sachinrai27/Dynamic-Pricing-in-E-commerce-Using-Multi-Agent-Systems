# Importing required libraries for mathematical operations, PyTorch, and random sampling.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import statistics

from dqnn import DQN
from replay_buffer import ReplayBuffer
from vendor_agent import VendorAgent
from product import Product
from marketplace import Marketplace

# Setting the seed for reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def analyze_results(marketplace):
    """
    Analyzes the results of the marketplace simulation.

    Args:
        marketplace (Marketplace): The marketplace after the simulation.

    """
    import seaborn as sns
    from scipy import stats

    # Collect history data from all vendors
    all_rewards = []
    all_actions = []
    for vendor in marketplace.vendors:
        all_rewards.extend(vendor.history['rewards'])
        all_actions.extend(vendor.history['actions'])

    # Create DataFrames for rewards and actions
    rewards_df = pd.DataFrame(all_rewards, columns=['reward'])
    actions_df = pd.DataFrame(all_actions, columns=['action'])

    # Plotting the distribution of rewards and actions
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    sns.histplot(data=rewards_df, x='reward', kde=True, ax=ax[0], bins=50)
    ax[0].set_title('Reward Distribution')
    
    sns.histplot(data=actions_df, x='action', kde=True, ax=ax[1], bins=50)
    ax[1].set_title('Action Distribution')

    plt.tight_layout()
    plt.show()

    # Compute and print various statistics for rewards
    print(f"Reward Mean: {rewards_df['reward'].mean()}")
    print(f"Reward Median: {rewards_df['reward'].median()}")
    print(f"Reward Standard Deviation: {rewards_df['reward'].std()}")
    print(f"Reward 95% Confidence Interval: {stats.norm.interval(0.95, loc=rewards_df['reward'].mean(), scale=rewards_df['reward'].std())}")

    # Compute and print various statistics for actions
    print(f"Action Mean: {actions_df['action'].mean()}")
    print(f"Action Median: {actions_df['action'].median()}")
    print(f"Action Standard Deviation: {actions_df['action'].std()}")
    print(f"Action 95% Confidence Interval: {stats.norm.interval(0.95, loc=actions_df['action'].mean(), scale=actions_df['action'].std())}")

    # Analyze average profits per episode
    avg_profits_df = pd.DataFrame(marketplace.history['average_profit'], columns=['average_profit'])

    # Plotting the average profit per episode
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=avg_profits_df, x=avg_profits_df.index, y='average_profit')
    plt.title('Average Profit per Episode Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Average Profit')
    plt.show()

    # Compute and print various statistics for average profits
    print(f"Average Profit Mean: {avg_profits_df['average_profit'].mean()}")
    print(f"Average Profit Median: {avg_profits_df['average_profit'].median()}")
    print(f"Average Profit Standard Deviation: {avg_profits_df['average_profit'].std()}")
    print(f"Average Profit 95% Confidence Interval: {stats.norm.interval(0.95, loc=avg_profits_df['average_profit'].mean(), scale=avg_profits_df['average_profit'].std())}")


def main():
    """
    Main function to run the marketplace simulation.

    This function initializes the products, vendors, and marketplace,
    and then runs the simulation for a specified number of episodes and time steps.
    It also calculates and prints various statistics and plots related to the simulation.

    Note: The specific values used for product parameters, base demand, and number of episodes/time steps
    are subject to change based on the desired simulation configuration.

    """
    # Initializing products for vendors
    products1 = [Product(100, 150, 100), Product(150, 200, 150), Product(200, 250, 200)]
    products2 = [Product(120, 170, 120), Product(160, 210, 160), Product(180, 230, 180)]

    # Setting base demands for vendors
    base_demand1 = 100
    base_demand2 = 120

    # Creating a list of VendorAgent objects with different configurations
    vendors = [VendorAgent(products1, 3, list(range(-i, i + 1)), base_demand1) for i in range(1, 11)]
    vendors += [VendorAgent(products2, 3, list(range(-i, i + 1)), base_demand2) for i in range(1, 11)]

    # Creating a marketplace object with the vendors
    marketplace = Marketplace(vendors)

    # Running the simulation for a specified number of episodes and time steps
    marketplace.simulate(episodes=1000, time_steps=10)

    # Total rewards obtained by each vendor
    vendor_rewards = [vendor.total_rewards for vendor in marketplace.vendors]

    # Rewards obtained by each vendor
    rewards = []
    for vendor in marketplace.vendors:
        rewards.extend(vendor.history['rewards'])

    # Actions taken by each vendor
    actions = []
    for vendor in marketplace.vendors:
        actions.extend(vendor.history['actions'])

    # Statistics related to total rewards
    print('Mean of total rewards:', statistics.mean(vendor_rewards))
    print('Median of total rewards:', statistics.median(vendor_rewards))
    print('Standard deviation of total rewards:', statistics.stdev(vendor_rewards))

    # Average profit per episode
    plt.plot(range(len(marketplace.history['average_profit'])), marketplace.history['average_profit'])
    plt.xlabel('Episode')
    plt.ylabel('Average Profit')
    plt.title('Average Profit per Episode Over Time')
    plt.show()

    # Converting the list of actions to a numpy array
    actions = np.array(actions)

    # Plotting a histogram of actions
    plt.hist(actions, bins=range(-10, 11), alpha=0.75, edgecolor='black')
    plt.title('Histogram of Actions')
    plt.xlabel('Action')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Converting the list of rewards to a numpy array
    rewards = np.array(rewards)

    # Calculating the mean and standard deviation of rewards
    mean_reward = np.mean(rewards)
    std_dev = np.std(rewards)

    # Calculating the 95% confidence interval for the mean reward
    confidence_interval = (mean_reward - 1.96 * std_dev / np.sqrt(len(rewards)),
                           mean_reward + 1.96 * std_dev / np.sqrt(len(rewards)))

    print(f"The mean reward per episode is {mean_reward} with a 95% confidence interval of {confidence_interval}.")
    
    analyze_results(marketplace)


if __name__ == '__main__':
    main()
