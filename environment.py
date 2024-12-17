# environment.py

import numpy as np
import random

class P2LifeEnv:
    def __init__(self, grid_size=4, max_generations=5, seed=None, verbose=False):
        """
        Initialize the P2Life environment.
        
        Parameters:
        - grid_size (int): Size of the grid (e.g., 4 for 4x4).
        - max_generations (int): Number of generations per game.
        - seed (int): Random seed for reproducibility.
        - verbose (bool): If True, prints the grid at each step.
        """
        self.grid_size = grid_size
        self.max_generations = max_generations
        self.seed = seed
        self.verbose = verbose
        self.players = ['A', 'B']  # Player A: White, Player B: Black
        self.reset()

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.generation = 0
        rng = np.random.RandomState(self.seed)
        # Initialize grid with random cells
        # 0: Dead, 1: White (Player A), 2: Black (Player B)
        self.grid = rng.choice([0, 1, 2], size=(self.grid_size, self.grid_size), p=[0.6, 0.2, 0.2])
        if self.verbose:
            self.print_grid()

    def get_valid_actions(self, player):
        """
        Get all valid actions for the given player.
        
        Parameters:
        - player (str): 'A' or 'B'
        
        Returns:
        - List of action indices (0 to grid_size^2 - 1)
        """
        valid_actions = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = self.grid[r, c]
                if cell == 0:
                    valid_actions.append(r * self.grid_size + c)
                elif (player == 'A' and cell == 2) or (player == 'B' and cell == 1):
                    valid_actions.append(r * self.grid_size + c)
        return valid_actions

    def step(self, actions):
        """
        Apply actions from both players and evolve the grid.
        
        Parameters:
        - actions (tuple): (action_A, action_B)
        
        Returns:
        - state (tuple): Current grid state as a tuple of tuples.
        - rewards (tuple): (reward_A, reward_B)
        - done (bool): Whether the game has ended.
        """
        action_A, action_B = actions
        # Apply Player A's action
        if action_A is not None:
            r, c = divmod(action_A, self.grid_size)
            self._toggle_cell(r, c, 'A')
        
        # Apply Player B's action
        if action_B is not None:
            r, c = divmod(action_B, self.grid_size)
            self._toggle_cell(r, c, 'B')
        
        # Evolve the grid
        self._evolve()
        
        # Calculate rewards
        reward_A = np.sum(self.grid == 1)
        reward_B = np.sum(self.grid == 2)
        
        # Increment generation
        self.generation += 1
        
        # Check termination
        done = False
        if self.generation >= self.max_generations or (reward_A + reward_B == 0):
            done = True
            if self.verbose:
                print("Game Over.")
                self.print_grid()
        
        if self.verbose:
            self.print_grid()
            print(f"Generation {self.generation} - Reward A: {reward_A}, Reward B: {reward_B}\n")
        
        # Convert grid to state tuple
        state = tuple(tuple(row) for row in self.grid)
        
        return state, (reward_A, reward_B), done

    def _toggle_cell(self, r, c, player):
        """
        Toggle a cell based on the player's action.
        
        Parameters:
        - r (int): Row index.
        - c (int): Column index.
        - player (str): 'A' or 'B'
        """
        if self.grid[r, c] == 0:
            self.grid[r, c] = 1 if player == 'A' else 2
        elif (player == 'A' and self.grid[r, c] == 2) or (player == 'B' and self.grid[r, c] == 1):
            self.grid[r, c] = 0

    def _count_neighbors(self, r, c):
        """
        Count the number of white and black neighbors for a cell.
        
        Parameters:
        - r (int): Row index.
        - c (int): Column index.
        
        Returns:
        - white (int): Number of white neighbors.
        - black (int): Number of black neighbors.
        """
        white = 0
        black = 0
        for i in range(r-1, r+2):
            for j in range(c-1, c+2):
                if i == r and j == c:
                    continue
                ni, nj = i % self.grid_size, j % self.grid_size
                if self.grid[ni, nj] == 1:
                    white += 1
                elif self.grid[ni, nj] == 2:
                    black += 1
        return white, black

    def _evolve(self):
        """
        Evolve the grid based on P2Life rules.
        """
        new_grid = self.grid.copy()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                white, black = self._count_neighbors(r, c)
                cell = self.grid[r, c]
                if cell == 1:  # White cell
                    diff = white - black
                    if diff in [2, 3] or (diff == 1 and white >= 2):
                        new_grid[r, c] = 1
                    else:
                        new_grid[r, c] = 0
                elif cell == 2:  # Black cell
                    diff = black - white
                    if diff in [2, 3] or (diff == 1 and black >= 2):
                        new_grid[r, c] = 2
                    else:
                        new_grid[r, c] = 0
                else:  # Dead cell
                    if white == 3 and black != 3:
                        new_grid[r, c] = 1
                    elif black == 3 and white != 3:
                        new_grid[r, c] = 2
                    elif white == 3 and black == 3:
                        new_grid[r, c] = random.choice([1, 2])
        self.grid = new_grid

    def get_state(self):
        """
        Get the current state.
        
        Returns:
        - state (tuple): Current grid state as a tuple of tuples.
        """
        return tuple(tuple(row) for row in self.grid)

    def print_grid(self):
        """
        Print the current grid to the terminal.
        """
        print(f"Generation {self.generation}:")
        for row in self.grid:
            print(' '.join(['W' if cell == 1 else 'B' if cell == 2 else '.' for cell in row]))
        print("-" * (2 * self.grid_size))
