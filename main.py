# =============================================================================
# MAIN EXECUTION PIPELINE
# Q-Learning for Credit Card Fraud Detection (Anomaly Detection)
# =============================================================================
# This implements the complete ML pipeline:
# 1. Data Loading & Preprocessing
# 2. Train/Test Split
# 3. Q-Learning Training
# 4. Evaluation on Test Set
# 5. Visualization of Results
# =============================================================================

from config.config import *
from data.loader import load_data, split_train_test
from environment.fraud_env import FraudEnvironment
from agent.q_learning_agent import QLearningAgent
from training.trainer import train
from visualization.plots import plot_rewards, plot_metrics
from evaluation.metrics import evaluate_agent, print_evaluation_results

# =============================================================================
# ALGORITHM BEHAVIOR EXPLANATION (CO3)
# =============================================================================
# Q-Learning Algorithm Flow:
#
# 1. INITIALIZATION:
#    - Q(s,a) = 0 for all state-action pairs (optimistic initialization)
#    - Agent starts with high exploration (ε = 1.0)
#
# 2. FOR EACH EPISODE (pass through dataset):
#    a) Reset environment to first transaction
#    b) FOR EACH TRANSACTION:
#       - Observe current state s (amount, time, risk features)
#       - Choose action a using ε-greedy policy:
#         * With probability ε: random action (explore)
#         * With probability (1-ε): best action (exploit)
#       - Execute action, observe reward r and next state s'
#       - Update Q-value: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
#       - Move to next transaction
#    c) Decay exploration: ε ← ε * 0.995
#
# 3. CONVERGENCE:
#    - Over time, Q-values converge to optimal action-value function Q*
#    - Agent learns which actions maximize long-term reward
#    - Exploration decreases, exploitation increases
#
# 4. EVALUATION:
#    - Use learned policy (greedy) on unseen test data
#    - No exploration, only exploitation of learned knowledge
# =============================================================================

print("="*70)
print(" Q-LEARNING FOR CREDIT CARD FRAUD DETECTION")
print("="*70)
print()

# =============================================================================
# STEP 1: DATA LOADING AND PREPROCESSING
# =============================================================================
print("STEP 1: Loading and preprocessing data...")
print("-"*70)

# Load dataset with optional balancing and size limiting
# balance=True helps with severely imbalanced fraud data (0.17% fraud)
# max_samples limits data size for faster experimentation
data = load_data(
    "creditcard.csv",
    balance=BALANCE_DATA,
    max_samples=MAX_SAMPLES
)

# Split into training and testing sets
# Training: Learn Q-values
# Testing: Evaluate learned policy on unseen data
train_data, test_data = split_train_test(data, TRAIN_TEST_SPLIT)

print()

# =============================================================================
# STEP 2: ENVIRONMENT AND AGENT INITIALIZATION
# =============================================================================
print("STEP 2: Initializing environment and agent...")
print("-"*70)

# Create training environment
# FraudEnvironment wraps the data and provides step-by-step interaction
train_env = FraudEnvironment(train_data, AMOUNT_BINS)

# Create Q-Learning agent
# actions=[0, 1]: 0=approve transaction, 1=flag as fraud
agent = QLearningAgent(
    actions=[0, 1],
    alpha=ALPHA,      # Learning rate: 0.1
    gamma=GAMMA,      # Discount factor: 0.95
    epsilon=EPSILON   # Initial exploration: 1.0 (100% random)
)

print(f"Environment: {len(train_data)} training transactions")
print(f"State space: {3} (amount) × {2} (time) × {2} (risk) = {12} states")
print(f"Action space: {len(agent.actions)} actions (approve/flag)")
print(f"Hyperparameters:")
print(f"  - Learning rate (α): {ALPHA}")
print(f"  - Discount factor (γ): {GAMMA}")
print(f"  - Initial exploration (ε): {EPSILON}")
print(f"  - Epsilon decay: {EPSILON_DECAY}")
print(f"  - Min epsilon: {MIN_EPSILON}")
print(f"  - Episodes: {EPISODES}")
print()

# =============================================================================
# STEP 3: TRAINING PHASE
# =============================================================================
print("STEP 3: Training Q-Learning agent...")
print("-"*70)
print(f"Running {EPISODES} episodes through training data...")
print("(This may take a few minutes depending on dataset size)")
print()

# Train the agent
# Returns list of total rewards per episode (learning curve)
rewards_history = train(
    agent=agent,
    env=train_env,
    episodes=EPISODES,
    epsilon_decay=EPSILON_DECAY,
    min_epsilon=MIN_EPSILON
)

print(f"\nTraining complete!")
print(f"Final exploration rate (ε): {agent.epsilon:.4f}")
print(f"Q-table size: {len(agent.q)} state-action pairs learned")
print()

# =============================================================================
# STEP 4: EVALUATION ON TEST SET
# =============================================================================
print("STEP 4: Evaluating on test set...")
print("-"*70)

# Create test environment with unseen data
test_env = FraudEnvironment(test_data, AMOUNT_BINS)

# Evaluate agent using learned policy (no exploration)
# This tests how well the agent generalizes to new transactions
test_metrics = evaluate_agent(agent, test_env, test_data)

# Display results
print_evaluation_results(test_metrics)

# =============================================================================
# STEP 5: VISUALIZATION
# =============================================================================
print("STEP 5: Generating visualizations...")
print("-"*70)

# Plot learning curve (rewards over episodes)
plot_rewards(rewards_history)

# Plot evaluation metrics
plot_metrics(test_metrics)

print("\nVisualization windows opened. Close them to continue.")
print()

# =============================================================================
# STEP 6: FINAL SUMMARY
# =============================================================================
print("="*70)
print(" EXPERIMENT SUMMARY")
print("="*70)
print(f"\nDataset:")
print(f"  - Total transactions: {len(data)}")
print(f"  - Training set: {len(train_data)}")
print(f"  - Test set: {len(test_data)}")
print(f"  - Fraud rate: {(data['Class'].sum() / len(data) * 100):.3f}%")

print(f"\nTraining:")
print(f"  - Episodes: {EPISODES}")
print(f"  - Final ε: {agent.epsilon:.4f}")
print(f"  - Learned states: {len(set([k[0] for k in agent.q.keys()]))} unique states")

print(f"\nTest Performance:")
print(f"  - Detection Rate: {test_metrics['detection_rate']*100:.2f}%")
print(f"  - Precision: {test_metrics['precision']*100:.2f}%")
print(f"  - F1 Score: {test_metrics['f1_score']:.4f}")
print(f"  - Accuracy: {test_metrics['accuracy']*100:.2f}%")

print("\n" + "="*70)
print("Experiment completed successfully!")
print("="*70)