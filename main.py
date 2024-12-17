# main.py

import time
import random
import numpy as np
from environment import P2LifeEnv
from agents import RandomAgent, MonteCarloAgent, QLearningAgent

def opponent_move(env, player_color):
    """
    Define the opponent's move strategy.
    Currently, the opponent is a random agent.
    
    Parameters:
    - env (P2LifeEnv): The game environment.
    - player_color (str): 'A' or 'B'
    
    Returns:
    - action (int): Cell index to toggle.
    """
    opponent = RandomAgent('B' if player_color == 'A' else 'A')
    return opponent.pick_move(env)

def train_mc(agent, env, episodes=500, horizon=5, verbose=False):
    """
    Train the Monte Carlo agent.
    
    Parameters:
    - agent (MonteCarloAgent): The Monte Carlo agent.
    - env (P2LifeEnv): The game environment.
    - episodes (int): Number of training episodes.
    - horizon (int): Number of generations per episode.
    - verbose (bool): If True, prints progress.
    
    Returns:
    - win_rates (list): Win rate over time.
    """
    win_rates = []
    for ep in range(1, episodes + 1):
        env.reset()
        episode = []
        for _ in range(horizon):
            state = env.get_state()
            action = agent.pick_move(env)
            # Player A acts
            agent_action = action
            # Opponent acts
            opponent_action = opponent_move(env, agent.player_color)
            # Apply both actions
            next_state, rewards, done = env.step((agent_action, opponent_action))
            reward_A, reward_B = rewards
            # Store transition
            episode.append((state, agent_action, reward_A))
            if done:
                break
        # Update agent after episode
        agent.update(episode)
        # Evaluate periodically
        if ep % 50 == 0:
            win_rate = evaluate_agent(agent, env, episodes=100, horizon=horizon)
            win_rates.append(win_rate)
            if verbose:
                print(f"Episode {ep}: Win Rate: {win_rate * 100:.2f}%")
    return win_rates

def train_q_learning(agent, env, episodes=500, horizon=5, verbose=False):
    """
    Train the Q-Learning agent.
    
    Parameters:
    - agent (QLearningAgent): The Q-Learning agent.
    - env (P2LifeEnv): The game environment.
    - episodes (int): Number of training episodes.
    - horizon (int): Number of generations per episode.
    - verbose (bool): If True, prints progress.
    
    Returns:
    - win_rates (list): Win rate over time.
    """
    win_rates = []
    for ep in range(1, episodes + 1):
        env.reset()
        for _ in range(horizon):
            state = env.get_state()
            action = agent.pick_move(env)
            # Player A acts
            agent_action = action
            # Opponent acts
            opponent_action = opponent_move(env, agent.player_color)
            # Apply both actions
            next_state, rewards, done = env.step((agent_action, opponent_action))
            reward_A, reward_B = rewards
            # Update agent
            agent.update(state, agent_action, reward_A, next_state, done, env)
            if done:
                break
        # Evaluate periodically
        if ep % 50 == 0:
            win_rate = evaluate_agent(agent, env, episodes=100, horizon=horizon)
            win_rates.append(win_rate)
            if verbose:
                print(f"Episode {ep}: Win Rate: {win_rate * 100:.2f}%")
    return win_rates

def evaluate_agent(agent, env, episodes=100, horizon=5):
    """
    Evaluate the agent against a random opponent.
    
    Parameters:
    - agent: The agent to evaluate.
    - env (P2LifeEnv): The game environment.
    - episodes (int): Number of evaluation episodes.
    - horizon (int): Number of generations per episode.
    
    Returns:
    - win_rate (float): Percentage of games the agent wins.
    """
    wins = 0
    for _ in range(episodes):
        env.reset()
        for _ in range(horizon):
            state = env.get_state()
            action = agent.pick_move(env)
            # Player A acts
            agent_action = action
            # Opponent acts
            opponent_action = opponent_move(env, agent.player_color)
            # Apply both actions
            next_state, rewards, done = env.step((agent_action, opponent_action))
            reward_A, reward_B = rewards
            if done:
                break
        # Determine winner
        if reward_A > reward_B:
            wins += 1
    return wins / episodes

def main():
    # Initialize environment
    env = P2LifeEnv(grid_size=4, max_generations=5, seed=42, verbose=False)
    
    # Initialize agents
    mc_agent = MonteCarloAgent(player_color='A', epsilon=0.1, gamma=0.9)
    ql_agent = QLearningAgent(player_color='A', alpha=0.1, gamma=0.9, epsilon=0.1)
    random_agent = RandomAgent(player_color='A')  # For baseline
    
    # Training parameters
    episodes = 500
    horizon = 5
    print_interval = 50
    
    print("Training Monte Carlo Agent...")
    mc_win_rates = train_mc(mc_agent, env, episodes=episodes, horizon=horizon, verbose=True)
    
    print("\nTraining Q-Learning Agent...")
    ql_win_rates = train_q_learning(ql_agent, env, episodes=episodes, horizon=horizon, verbose=True)
    
    # Final Evaluation
    print("\nFinal Evaluation:")
    mc_final_win_rate = evaluate_agent(mc_agent, env, episodes=100, horizon=horizon)
    ql_final_win_rate = evaluate_agent(ql_agent, env, episodes=100, horizon=horizon)
    random_final_win_rate = evaluate_agent(random_agent, env, episodes=100, horizon=horizon)
    
    print(f"Monte Carlo Agent Win Rate: {mc_final_win_rate * 100:.2f}%")
    print(f"Q-Learning Agent Win Rate: {ql_final_win_rate * 100:.2f}%")
    print(f"Random Agent Win Rate: {random_final_win_rate * 100:.2f}%")

if __name__ == "__main__":
    main()
