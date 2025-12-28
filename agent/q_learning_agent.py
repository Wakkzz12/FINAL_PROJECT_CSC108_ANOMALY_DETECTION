import random
from collections import defaultdict

# =============================================================================
# Q-LEARNING AGENT FOR FRAUD DETECTION
# =============================================================================
# Implements the Q-Learning algorithm (Watkins, 1989) for binary classification
# of credit card transactions as legitimate (0) or fraudulent (1).
#
# TIME COMPLEXITY ANALYSIS:
# - choose_action(): O(|A|) where |A| = number of actions (2 in this case)
# - update(): O(|A|) for finding max Q-value over next actions
# - Per episode: O(N * |A|) where N = number of transactions
# 
# SPACE COMPLEXITY ANALYSIS:
# - Q-table storage: O(|S| * |A|) where |S| = number of unique states
# - With discretization: |S| = 3 (amount) * 2 (time) * 2 (risk) = 12 states
# - Total space: O(12 * 2) = O(24) = O(1) constant space for Q-table
# =============================================================================

class QLearningAgent:
    """
    Q-Learning Agent implementing off-policy TD control.
    
    The Q-Learning update rule (Bellman Equation):
    Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
    
    Where:
    - Q(s,a): Action-value function (expected return from state s taking action a)
    - α: Learning rate (step size)
    - r: Immediate reward
    - γ: Discount factor
    - s': Next state
    - max_a' Q(s',a'): Maximum Q-value over all actions in next state
    """
    
    def __init__(self, actions, alpha, gamma, epsilon):
        """
        Initialize Q-Learning agent.
        
        Args:
            actions (list): Available actions [0=approve, 1=flag as fraud]
            alpha (float): Learning rate (0 to 1)
            gamma (float): Discount factor (0 to 1)
            epsilon (float): Initial exploration rate (0 to 1)
        """
        # Q-table: Maps (state, action) pairs to Q-values
        # Using defaultdict(float) initializes all Q-values to 0
        # This is the optimistic initialization strategy
        self.q = defaultdict(float)
        
        self.actions = actions  # [0, 1]
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate (decays over time)
    
    def choose_action(self, state):
        """
        Epsilon-greedy action selection policy.
        
        With probability epsilon: explore (random action)
        With probability (1-epsilon): exploit (best known action)
        
        This balances exploration vs exploitation, crucial for Q-Learning
        to discover optimal policy while also leveraging learned knowledge.
        
        Time Complexity: O(|A|) = O(2) = O(1)
        - O(1) for random selection
        - O(|A|) for finding argmax over actions
        
        Args:
            state (tuple): Current state representation
            
        Returns:
            int: Selected action (0 or 1)
        """
        # EXPLORATION: Random action with probability epsilon
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        # EXPLOITATION: Greedy action selection (argmax)
        # Choose action with highest Q-value for current state
        return max(self.actions, key=lambda a: self.q[(state, a)])
    
    def update(self, state, action, reward, next_state):
        """
        Q-Learning update rule (Bellman equation for optimal policy).
        
        This is the core of the Q-Learning algorithm. It updates the Q-value
        for the (state, action) pair based on:
        1. The immediate reward received
        2. The estimated future value (discounted max Q-value of next state)
        3. The current Q-value estimate
        
        Time Complexity: O(|A|) = O(2) = O(1)
        - Finding max over actions in next_state
        
        Args:
            state (tuple): Previous state
            action (int): Action taken
            reward (float): Immediate reward received
            next_state (tuple or None): Resulting state (None if terminal)
        """
        # Calculate TD target: r + γ·max_a' Q(s',a')
        # If next_state is None (terminal state), future value is 0
        if next_state is not None:
            # Find maximum Q-value over all possible actions in next state
            # This represents the value of the best action we could take next
            max_next_q = max([self.q[(next_state, a)] for a in self.actions])
        else:
            # Terminal state: no future rewards possible
            max_next_q = 0
        
        # Current Q-value estimate
        current_q = self.q[(state, action)]
        
        # TD error: how much our estimate was off
        # td_error = (r + γ·max_a' Q(s',a')) - Q(s,a)
        td_error = reward + self.gamma * max_next_q - current_q
        
        # Update Q-value using gradient descent step
        # Q(s,a) ← Q(s,a) + α·[TD_target - Q(s,a)]
        self.q[(state, action)] = current_q + self.alpha * td_error
    
    def get_greedy_action(self, state):
        """
        Get best action without exploration (pure exploitation).
        Used during evaluation phase.
        
        Args:
            state (tuple): Current state
            
        Returns:
            int: Best action according to learned policy
        """
        return max(self.actions, key=lambda a: self.q[(state, a)])
    
    def get_q_value(self, state, action):
        """
        Retrieve Q-value for state-action pair.
        Useful for debugging and visualization.
        
        Args:
            state (tuple): State
            action (int): Action
            
        Returns:
            float: Q-value
        """
        return self.q[(state, action)]