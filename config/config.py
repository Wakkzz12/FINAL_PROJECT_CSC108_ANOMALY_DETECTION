# =============================================================================
# Q-LEARNING HYPERPARAMETERS
# =============================================================================
ALPHA = 0.1           # Learning rate
GAMMA = 0.95          # Discount factor
EPSILON = 1.0         # Initial exploration rate
EPSILON_DECAY = 0.995 # Decay rate per episode
MIN_EPSILON = 0.05    # Minimum exploration rate

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
EPISODES = 50  # ← INCREASED from 30 to 50 for better convergence

ANOMALY_PENALTY = 10  # Not used currently

# =============================================================================
# STATE DISCRETIZATION BINS
# =============================================================================
AMOUNT_BINS = [0, 50, 200, 1000]

# =============================================================================
# EVALUATION PARAMETERS
# =============================================================================
TRAIN_TEST_SPLIT = 0.8

# IMPORTANT: Set to False to use more data!
BALANCE_DATA = False  # ← CHANGED: Use full imbalanced dataset

# Use 50,000 samples (good balance of speed vs accuracy)
MAX_SAMPLES = 50000  # ← CHANGED: Use much more data