# test.py

import time
from environment import P2LifeEnv
from agents import MonteCarloAgent, QLearningAgent, RandomAgent

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

def run_test():
    """
    Run a test scenario for P2Life with RL agents.
    
    Description:
    This test evaluates how well a Q-Learning agent and a Monte Carlo agent perform against a Random opponent
    in a simplified two-player variant of Conway's Game of Life on a 4x4 grid over 5 generations.
    
    Research Question:
    How effectively can Q-Learning and Monte Carlo methods learn to control more cells than a random opponent
    in a competitive cellular automaton environment?
    
    Results:
    After training, both RL agents are evaluated over 100 games each. The win rates are displayed to compare
    their performance against the Random baseline.
    """
    # Initialize environment
    env = P2LifeEnv(grid_size=4, max_generations=5, seed=42, verbose=False)
    
    # Initialize agents
    mc_agent = MonteCarloAgent(player_color='A', epsilon=0.1, gamma=0.9)
    ql_agent = QLearningAgent(player_color='A', alpha=0.1, gamma=0.9, epsilon=0.1)
    random_agent = RandomAgent(player_color='A')  # For baseline
    
    # Training parameters
    episodes = 100  # Reduced for quick testing
    horizon = 5
    
    print("Training Monte Carlo Agent...")
    # Train Monte Carlo Agent
    for ep in range(1, episodes + 1):
        env.reset()
        episode = []
        for _ in range(horizon):
            state = env.get_state()
            action = mc_agent.pick_move(env)
            # Player A acts
            agent_action = action
            # Opponent acts
            opponent_action = opponent_move(env, mc_agent.player_color)
            # Apply both actions
            next_state, rewards, done = env.step((agent_action, opponent_action))
            reward_A, reward_B = rewards
            # Store transition
            episode.append((state, agent_action, reward_A))
            if done:
                break
        # Update agent after episode
        mc_agent.update(episode)
    
    print("Training Q-Learning Agent...")
    # Train Q-Learning Agent
    for ep in range(1, episodes + 1):
        env.reset()
        for _ in range(horizon):
            state = env.get_state()
            action = ql_agent.pick_move(env)
            # Player A acts
            agent_action = action
            # Opponent acts
            opponent_action = opponent_move(env, ql_agent.player_color)
            # Apply both actions
            next_state, rewards, done = env.step((agent_action, opponent_action))
            reward_A, reward_B = rewards
            # Update agent
            ql_agent.update(state, agent_action, reward_A, next_state, done, env)
            if done:
                break
    
    # Final Evaluation
    print("\nFinal Evaluation:")
    mc_final_win_rate = evaluate_agent(mc_agent, env, episodes=100, horizon=horizon)
    ql_final_win_rate = evaluate_agent(ql_agent, env, episodes=100, horizon=horizon)
    random_final_win_rate = evaluate_agent(random_agent, env, episodes=100, horizon=horizon)
    
    print(f"Monte Carlo Agent Win Rate: {mc_final_win_rate * 100:.2f}%")
    print(f"Q-Learning Agent Win Rate: {ql_final_win_rate * 100:.2f}%")
    print(f"Random Agent Win Rate: {random_final_win_rate * 100:.2f}%")
    
    # Display results summary
    print("\nTest Summary:")
    print("===================================")
    print("Reinforcement Learning in Two-Player P2Life")
    print("-----------------------------------")
    print("Objective: Determine how well Q-Learning and Monte Carlo agents can control more cells")
    print("          than a random opponent in a competitive cellular automaton environment.")
    print("\nResults over 100 evaluation games:")
    print(f"Monte Carlo Agent: {mc_final_win_rate * 100:.2f}% wins")
    print(f"Q-Learning Agent: {ql_final_win_rate * 100:.2f}% wins")
    print(f"Random Agent: {random_final_win_rate * 100:.2f}% wins")
    print("===================================")

if __name__ == "__main__":
    run_test()
