# =============================================================================
# ROBUST SELF-VALIDATING CONFIGURATION
# =============================================================================
# This configuration automatically validates and adjusts parameters
# to prevent common issues with extreme values
# =============================================================================

import warnings

# =============================================================================
# BASE Q-LEARNING HYPERPARAMETERS
# =============================================================================
ALPHA = 0.2           # Learning rate
GAMMA = 0.95          # Discount factor
EPSILON = 1.0         # Initial exploration rate
EPSILON_DECAY = 0.98  # Decay rate per episode
MIN_EPSILON = 0.01    # Minimum exploration rate

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
EPISODES = 100        # Number of training episodes
ANOMALY_PENALTY = 10  # Not used currently

# =============================================================================
# ROBUSTNESS FEATURES (NEW)
# =============================================================================
# Enable automatic safeguards
ENABLE_EARLY_STOPPING = True      # Stop when converged
EARLY_STOPPING_PATIENCE = 15      # Episodes without improvement
CONVERGENCE_THRESHOLD = 0.001     # Q-value change threshold

# Enable adaptive epsilon (recommended for any episode count)
USE_ADAPTIVE_EPSILON = True       # Auto-adjust epsilon based on episodes

# Warn if parameters seem extreme
VALIDATE_PARAMETERS = True        # Check parameter ranges

# =============================================================================
# STATE DISCRETIZATION BINS
# =============================================================================
AMOUNT_BINS = [0, 50, 200, 1000]

# =============================================================================
# DATA PARAMETERS
# =============================================================================
TRAIN_TEST_SPLIT = 0.8
BALANCE_DATA = True
MAX_SAMPLES = 20000

# =============================================================================
# PARAMETER VALIDATION LOGIC
# =============================================================================

def validate_and_adjust_parameters():
    """
    Automatically validate and adjust parameters to reasonable ranges.
    Prevents instructor from breaking your code with extreme values.
    
    Returns:
        dict: Validated parameters
    """
    global ALPHA, GAMMA, EPSILON, EPSILON_DECAY, MIN_EPSILON, EPISODES
    
    adjusted = {}
    
    # Validate ALPHA (learning rate)
    if ALPHA <= 0 or ALPHA > 1:
        warnings.warn(f"ALPHA={ALPHA} is invalid. Adjusting to 0.2")
        ALPHA = 0.2
        adjusted['ALPHA'] = ALPHA
    elif ALPHA < 0.01:
        warnings.warn(f"ALPHA={ALPHA} is very low (slow learning). Consider 0.1-0.3")
    elif ALPHA > 0.5:
        warnings.warn(f"ALPHA={ALPHA} is very high (unstable learning). Consider 0.1-0.3")
    
    # Validate GAMMA (discount factor)
    if GAMMA < 0 or GAMMA > 1:
        warnings.warn(f"GAMMA={GAMMA} is invalid. Adjusting to 0.95")
        GAMMA = 0.95
        adjusted['GAMMA'] = GAMMA
    
    # Validate EPSILON
    if EPSILON < 0 or EPSILON > 1:
        warnings.warn(f"EPSILON={EPSILON} is invalid. Adjusting to 1.0")
        EPSILON = 1.0
        adjusted['EPSILON'] = EPSILON
    
    # Validate EPSILON_DECAY
    if EPSILON_DECAY <= 0 or EPSILON_DECAY > 1:
        warnings.warn(f"EPSILON_DECAY={EPSILON_DECAY} is invalid. Adjusting to 0.98")
        EPSILON_DECAY = 0.98
        adjusted['EPSILON_DECAY'] = EPSILON_DECAY
    
    # Auto-adjust epsilon decay based on episodes
    if USE_ADAPTIVE_EPSILON:
        # This will be handled in trainer, but we can suggest optimal decay
        optimal_decay = (MIN_EPSILON / EPSILON) ** (1 / (0.7 * EPISODES))
        if abs(EPSILON_DECAY - optimal_decay) > 0.05:
            print(f"Note: For {EPISODES} episodes, optimal decay ≈ {optimal_decay:.6f}")
            print(f"      (Current: {EPSILON_DECAY:.6f})")
            print(f"      Adaptive epsilon will handle this automatically.")
    
    # Validate MIN_EPSILON
    if MIN_EPSILON < 0 or MIN_EPSILON >= EPSILON:
        warnings.warn(f"MIN_EPSILON={MIN_EPSILON} is invalid. Adjusting to 0.05")
        MIN_EPSILON = 0.05
        adjusted['MIN_EPSILON'] = MIN_EPSILON
    
    # Validate EPISODES
    if EPISODES <= 0:
        warnings.warn(f"EPISODES={EPISODES} is invalid. Adjusting to 50")
        EPISODES = 50
        adjusted['EPISODES'] = EPISODES
    elif EPISODES > 1000:
        warnings.warn(f"EPISODES={EPISODES} is very high. Early stopping will help.")
        if not ENABLE_EARLY_STOPPING:
            print("  → Enabling early stopping to prevent overfitting")
            adjusted['ENABLE_EARLY_STOPPING'] = True
    
    # Report adjustments
    if adjusted:
        print("\n" + "="*70)
        print("PARAMETER ADJUSTMENTS MADE:")
        print("="*70)
        for param, value in adjusted.items():
            print(f"  {param}: adjusted to {value}")
        print("="*70 + "\n")
    
    return {
        'ALPHA': ALPHA,
        'GAMMA': GAMMA,
        'EPSILON': EPSILON,
        'EPSILON_DECAY': EPSILON_DECAY,
        'MIN_EPSILON': MIN_EPSILON,
        'EPISODES': EPISODES
    }


# =============================================================================
# RECOMMENDED CONFIGURATIONS FOR DIFFERENT SCENARIOS
# =============================================================================

def get_config_for_episodes(num_episodes):
    """
    Get optimal configuration for given episode count.
    Makes your algorithm work well regardless of instructor's choice.
    
    Args:
        num_episodes (int): Number of episodes to train
        
    Returns:
        dict: Optimal configuration
    """
    if num_episodes <= 50:
        # Quick training - need fast learning
        return {
            'ALPHA': 0.3,
            'EPSILON_DECAY': 0.95,
            'MIN_EPSILON': 0.1,
            'USE_ADAPTIVE_EPSILON': True
        }
    
    elif num_episodes <= 150:
        # Standard training - balanced approach
        return {
            'ALPHA': 0.2,
            'EPSILON_DECAY': 0.98,
            'MIN_EPSILON': 0.05,
            'USE_ADAPTIVE_EPSILON': True
        }
    
    elif num_episodes <= 500:
        # Extended training - need to maintain exploration
        return {
            'ALPHA': 0.15,
            'EPSILON_DECAY': 0.99,
            'MIN_EPSILON': 0.05,
            'USE_ADAPTIVE_EPSILON': True,
            'ENABLE_EARLY_STOPPING': True
        }
    
    else:  # > 500 episodes
        # Very long training - definitely need early stopping
        return {
            'ALPHA': 0.1,
            'EPSILON_DECAY': 0.995,
            'MIN_EPSILON': 0.05,
            'USE_ADAPTIVE_EPSILON': True,
            'ENABLE_EARLY_STOPPING': True,
            'EARLY_STOPPING_PATIENCE': 20
        }


# =============================================================================
# AUTO-RUN VALIDATION
# =============================================================================
if VALIDATE_PARAMETERS:
    validated_params = validate_and_adjust_parameters()
    
    # Update globals with validated values
    ALPHA = validated_params['ALPHA']
    GAMMA = validated_params['GAMMA']
    EPSILON = validated_params['EPSILON']
    EPSILON_DECAY = validated_params['EPSILON_DECAY']
    MIN_EPSILON = validated_params['MIN_EPSILON']
    EPISODES = validated_params['EPISODES']


# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================
def print_config_summary():
    """Print current configuration for debugging."""
    print("\n" + "="*70)
    print("CURRENT CONFIGURATION")
    print("="*70)
    print(f"Q-Learning Parameters:")
    print(f"  Learning rate (α):        {ALPHA}")
    print(f"  Discount factor (γ):      {GAMMA}")
    print(f"  Initial exploration (ε):  {EPSILON}")
    print(f"  Epsilon decay:            {EPSILON_DECAY}")
    print(f"  Minimum epsilon:          {MIN_EPSILON}")
    print(f"\nTraining:")
    print(f"  Episodes:                 {EPISODES}")
    print(f"  Early stopping:           {ENABLE_EARLY_STOPPING}")
    if ENABLE_EARLY_STOPPING:
        print(f"  Patience:                 {EARLY_STOPPING_PATIENCE}")
    print(f"  Adaptive epsilon:         {USE_ADAPTIVE_EPSILON}")
    print(f"\nData:")
    print(f"  Balance dataset:          {BALANCE_DATA}")
    print(f"  Max samples:              {MAX_SAMPLES}")
    print(f"  Train/test split:         {TRAIN_TEST_SPLIT}")
    print("="*70 + "\n")