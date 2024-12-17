# agents.py

import random
from collections import defaultdict

class RandomAgent:
    def __init__(self, player_color):
        """
        Initialize a random agent.
        
        Parameters:
        - player_color (str): 'A' or 'B'
        """
        self.player_color = player_color

    def pick_move(self, env):
        """
        Pick a random valid move.
        
        Parameters:
        - env (P2LifeEnv): The game environment.
        
        Returns:
        - action (int): Cell index to toggle.
        """
        valid_actions = env.get_valid_actions(self.player_color)
        if not valid_actions:
            return None
        return random.choice(valid_actions)

class MonteCarloAgent:
    def __init__(self, player_color, epsilon=0.1, gamma=0.9):
        """
        Initialize a Monte Carlo agent.
        
        Parameters:
        - player_color (str): 'A' or 'B'
        - epsilon (float): Exploration rate.
        - gamma (float): Discount factor.
        """
        self.player_color = player_color
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(float)  # Q-values
        self.returns = defaultdict(list)  # Returns for each state-action
        self.policy = {}  # Policy derived from Q

    def pick_move(self, env):
        """
        Pick a move based on epsilon-greedy policy.
        
        Parameters:
        - env (P2LifeEnv): The game environment.
        
        Returns:
        - action (int): Cell index to toggle.
        """
        state = env.get_state()
        valid_actions = env.get_valid_actions(self.player_color)
        if not valid_actions:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Greedy action
        q_vals = [self.Q[(state, a)] for a in valid_actions]
        max_q = max(q_vals)
        best_actions = [a for a, q in zip(valid_actions, q_vals) if q == max_q]
        return random.choice(best_actions)

    def update(self, episode):
        """
        Update Q-values based on the episode.
        
        Parameters:
        - episode (list): List of (state, action, reward) tuples.
        """
        G = 0
        visited = set()
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            if (state, action) not in visited:
                self.returns[(state, action)].append(G)
                self.Q[(state, action)] = sum(self.returns[(state, action)]) / len(self.returns[(state, action)])
                visited.add((state, action))

class QLearningAgent:
    def __init__(self, player_color, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Initialize a Q-Learning agent.
        
        Parameters:
        - player_color (str): 'A' or 'B'
        - alpha (float): Learning rate.
        - gamma (float): Discount factor.
        - epsilon (float): Exploration rate.
        """
        self.player_color = player_color
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(float)  # Q-values

    def pick_move(self, env):
        """
        Pick a move based on epsilon-greedy policy.
        
        Parameters:
        - env (P2LifeEnv): The game environment.
        
        Returns:
        - action (int): Cell index to toggle.
        """
        state = env.get_state()
        valid_actions = env.get_valid_actions(self.player_color)
        if not valid_actions:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Greedy action
        q_vals = [self.Q[(state, a)] for a in valid_actions]
        max_q = max(q_vals)
        best_actions = [a for a, q in zip(valid_actions, q_vals) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done, env):
        """
        Update Q-values based on the transition.
        
        Parameters:
        - state (tuple): Current state.
        - action (int): Action taken.
        - reward (float): Reward received.
        - next_state (tuple): Next state after action.
        - done (bool): Whether the episode has ended.
        - env (P2LifeEnv): The game environment.
        """
        if done:
            target = reward
        else:
            valid_actions = env.get_valid_actions(self.player_color)
            if valid_actions:
                target = reward + self.gamma * max([self.Q[(next_state, a)] for a in valid_actions])
            else:
                target = reward
        
        self.Q[(state, action)] += self.alpha * (target - self.Q[(state, action)])
