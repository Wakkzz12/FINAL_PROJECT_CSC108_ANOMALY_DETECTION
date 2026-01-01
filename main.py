# =============================================================================
# MAIN EXECUTION PIPELINE - ROBUST VERSION
# Q-Learning for Credit Card Fraud Detection
# 
# This implementation is robust to extreme parameter values and
# automatically adjusts to prevent common issues.
# =============================================================================

from config.config import *
from data.loader import load_data  # Removed split_train_test
from environment.fraud_env import FraudEnvironment
from agent.q_learning_agent import QLearningAgent
from training.trainer import train_robust  # Use robust trainer
from visualization.plots import plot_rewards, plot_metrics
from evaluation.metrics import evaluate_agent, print_evaluation_results

print("="*70)
print(" Q-LEARNING FOR CREDIT CARD FRAUD DETECTION")
print(" (ROBUST IMPLEMENTATION)")
print("="*70)
print()

# =============================================================================
# STEP 0: CONFIGURATION VALIDATION
# =============================================================================
print("STEP 0: Validating configuration...")
print("-"*70)
print_config_summary()

# Auto-adjust configuration if episodes seem extreme
if EPISODES > 200:
    print(f"⚠️  High episode count detected ({EPISODES})")
    print(f"   Automatically enabling:")
    print(f"   - Early stopping (prevents overfitting)")
    print(f"   - Adaptive epsilon (maintains exploration)")
    print()
    ENABLE_EARLY_STOPPING = True
    USE_ADAPTIVE_EPSILON = True

# =============================================================================
# STEP 1: DATA LOADING AND PREPROCESSING
# =============================================================================
print("STEP 1: Loading and preprocessing data...")
print("-"*70)

data = load_data(
    "creditcard.csv",
    balance=BALANCE_DATA,
    max_samples=MAX_SAMPLES
)

# MODIFIED: Use the exact same data for training and testing
# Bypassing split_train_test as requested
train_data = data
test_data = data
print(f"Training set: {len(train_data)} transactions (100% of data)")
print(f"Testing set: {len(test_data)} transactions (100% of data)")
print()

# =============================================================================
# STEP 2: ENVIRONMENT AND AGENT INITIALIZATION
# =============================================================================
print("STEP 2: Initializing environment and agent...")
print("-"*70)

train_env = FraudEnvironment(train_data, AMOUNT_BINS)

agent = QLearningAgent(
    actions=[0, 1],
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=EPSILON
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
# STEP 3: ROBUST TRAINING PHASE
# =============================================================================
print("STEP 3: Training Q-Learning agent (with robustness features)...")
print("-"*70)

# Use robust training that handles any episode count
training_results = train_robust(
    agent=agent,
    env=train_env,
    episodes=EPISODES,
    epsilon_decay=EPSILON_DECAY if not USE_ADAPTIVE_EPSILON else None,
    min_epsilon=MIN_EPSILON,
    early_stopping=ENABLE_EARLY_STOPPING,
    adaptive_epsilon=USE_ADAPTIVE_EPSILON
)

rewards_history = training_results['rewards']
final_episode = training_results.get('final_episode', EPISODES)

print(f"\nTraining complete!")
print(f"Actual episodes trained: {final_episode}/{EPISODES}")
if training_results.get('stopped_early', False):
    print(f"  (Stopped early - Q-values converged)")
print(f"Final exploration rate (ε): {agent.epsilon:.4f}")
print(f"Q-table size: {len(agent.q)} state-action pairs learned")

# Check if training was successful
if len(rewards_history) < 10:
    print("\n WARNING: Very few episodes completed!")
    print("   This may indicate:")
    print("   - Dataset too small")
    print("   - Epsilon decay too aggressive")
    print("   Continuing with evaluation...")
elif final_episode < EPISODES * 0.3:
    print(f"\n✓ Converged quickly ({final_episode} episodes)")
    print("  This is good - agent learned efficiently!")

print()

# =============================================================================
# STEP 4: EVALUATION ON TEST SET
# =============================================================================
print("STEP 4: Evaluating on test set...")
print("-"*70)

test_env = FraudEnvironment(test_data, AMOUNT_BINS)
test_metrics = evaluate_agent(agent, test_env, test_data)

print_evaluation_results(test_metrics)

# Additional robustness check
if test_metrics['f1_score'] < 0.5:
    print("  WARNING: Low F1 score detected!")
    print("   Possible causes:")
    print("   - Insufficient training (try more episodes)")
    print("   - Dataset too small (increase MAX_SAMPLES)")
    print("   - Reward function needs tuning")
    print()

# =============================================================================
# STEP 5: VISUALIZATION
# =============================================================================
print("STEP 5: Generating visualizations...")
print("-"*70)

plot_rewards(rewards_history)
plot_metrics(test_metrics)

print("\nVisualization windows opened. Close them to continue.")
print()

# =============================================================================
# STEP 6: FINAL SUMMARY WITH QUALITY ASSESSMENT
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
print(f"  - Planned episodes: {EPISODES}")
print(f"  - Actual episodes: {final_episode}")
print(f"  - Early stopping: {'Yes' if training_results.get('stopped_early') else 'No'}")
print(f"  - Final ε: {agent.epsilon:.4f}")
print(f"  - Learned states: {len(set([k[0] for k in agent.q.keys()]))} unique states")

print(f"\nTest Performance:")
print(f"  - Detection Rate: {test_metrics['detection_rate']*100:.2f}%")
print(f"  - Precision: {test_metrics['precision']*100:.2f}%")
print(f"  - F1 Score: {test_metrics['f1_score']:.4f}")
print(f"  - Accuracy: {test_metrics['accuracy']*100:.2f}%")

# Quality assessment
print(f"\nQuality Assessment:")
f1 = test_metrics['f1_score']
if f1 >= 0.80:
    quality = "EXCELLENT"
elif f1 >= 0.70:
    quality = "GOOD"
elif f1 >= 0.60:
    quality = "ACCEPTABLE "
else:
    quality = "NEEDS IMPROVEMENT"

print(f"  Overall Quality: {quality}")

# Performance category
detection = test_metrics['detection_rate']
precision = test_metrics['precision']
fpr = test_metrics['false_positive_rate']

if detection >= 0.85 and precision >= 0.70 and fpr <= 0.35:
    category = "Production-Ready/High-Quality"
elif detection >= 0.80 and precision >= 0.60:
    category = "Research-Quality/Prototype"
else:
    category = "Experimental"

print(f"  Performance Category: {category}")

# Recommendations
print(f"\nRecommendations:")
if f1 < 0.70:
    print("  - Consider increasing training episodes or MAX_SAMPLES")
    print("  - Try adjusting reward function penalties")
elif fpr > 0.40:
    print("  - Increase false alarm penalty in reward function")
    print("  - Consider more conservative policy (higher MIN_EPSILON)")
elif detection < 0.80:
    print("  - Increase missed fraud penalty in reward function")
    print("  - Ensure sufficient training episodes")
else:
    print("  - Performance is excellent! No major adjustments needed.")
    print("  - For minor improvements, consider ensemble methods")

print("\n" + "="*70)
print("Experiment completed successfully!")
print("="*70)

# =============================================================================
# STEP 7: EXPORT RESULTS (OPTIONAL)
# =============================================================================
# Save results for later analysis
import json
from datetime import datetime

results_summary = {
    'timestamp': datetime.now().isoformat(),
    'configuration': {
        'alpha': ALPHA,
        'gamma': GAMMA,
        'episodes_planned': EPISODES,
        'episodes_actual': final_episode,
        'max_samples': MAX_SAMPLES
    },
    'performance': {
        'detection_rate': float(test_metrics['detection_rate']),
        'precision': float(test_metrics['precision']),
        'f1_score': float(test_metrics['f1_score']),
        'accuracy': float(test_metrics['accuracy']),
        'fpr': float(test_metrics['false_positive_rate'])
    },
    'quality': quality,
    'category': category
}

# Optionally save to file
try:
    with open('results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print("\nResults saved to 'results_summary.json'")
except Exception as e:
    print(f"\nNote: Could not save results file: {e}")