# =============================================================================
# Q-LEARNING HYPERPARAMETERS
# =============================================================================
# ALPHA: Learning rate (0 to 1)
# - Controls how much new information overrides old information
# - 0 = no learning, 1 = only consider most recent information
# - Typically set between 0.01 and 0.5 for stable convergence
ALPHA = 0.1

# GAMMA: Discount factor (0 to 1)
# - Determines importance of future rewards vs immediate rewards
# - 0 = only consider immediate rewards, 1 = future rewards equally important
# - 0.95 means future rewards are worth 95% of immediate rewards
GAMMA = 0.95

# EPSILON: Exploration rate (0 to 1)
# - Probability of taking random action vs best known action
# - Starts high (1.0) to explore, decays to exploit learned policy
EPSILON = 1.0

# EPSILON_DECAY: Rate of exploration decay per episode
# - Multiplied by epsilon after each episode
# - 0.995 means epsilon decreases by 0.5% per episode
EPSILON_DECAY = 0.995

# MIN_EPSILON: Minimum exploration rate
# - Ensures agent continues some exploration even after training
# - 0.05 = 5% random actions to avoid getting stuck in local optima
MIN_EPSILON = 0.05

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
# Number of complete passes through the dataset
# Note: With 284,807 transactions, this is a large number of state transitions
EPISODES = 30

# Not used in current implementation - can be removed or repurposed
ANOMALY_PENALTY = 10

# =============================================================================
# STATE DISCRETIZATION BINS
# =============================================================================
# Transaction amount thresholds for state space discretization
# Creates 3 bins: LOW (0-50), MEDIUM (50-200), HIGH (200+)
# Reduces continuous state space to manageable discrete states
AMOUNT_BINS = [0, 50, 200, 1000]

# =============================================================================
# EVALUATION PARAMETERS (NEW)
# =============================================================================
# Split ratio for training/testing
TRAIN_TEST_SPLIT = 0.8

# Whether to balance the dataset (important for imbalanced fraud data)
BALANCE_DATA = True

# Maximum samples to use (for faster testing, set to None for full dataset)
MAX_SAMPLES = 10000  # Set to None to use full dataset