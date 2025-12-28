import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
# Provides visual analysis of Q-Learning training and evaluation results
# =============================================================================

def plot_rewards(rewards):
    """
    Plot learning curve: total reward per episode over training.
    
    The learning curve shows:
    - How quickly the agent learns (steepness of curve)
    - Whether learning converges (curve flattens)
    - Overall performance trend (upward = improving)
    
    Expected behavior:
    - Initial episodes: Low/negative rewards (random exploration)
    - Middle episodes: Rapid improvement (learning patterns)
    - Final episodes: Plateaus (converged to good policy)
    
    Time Complexity: O(E) where E = number of episodes
    
    Args:
        rewards (list): Total reward per episode
    """
    plt.figure(figsize=(10, 6))
    
    # Plot raw rewards
    plt.plot(rewards, alpha=0.6, label='Episode Reward', linewidth=1)
    
    # Add moving average for trend
    if len(rewards) >= 5:
        window_size = min(10, len(rewards) // 3)
        moving_avg = np.convolve(
            rewards, 
            np.ones(window_size)/window_size, 
            mode='valid'
        )
        plt.plot(
            range(window_size-1, len(rewards)), 
            moving_avg, 
            'r-', 
            linewidth=2,
            label=f'{window_size}-Episode Moving Average'
        )
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.title("Q-Learning Training: Learning Curve", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add reference line at y=0
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_metrics(metrics):
    """
    Visualize evaluation metrics with multiple subplots.
    
    Creates a comprehensive dashboard showing:
    1. Confusion matrix (heatmap)
    2. Key performance metrics (bar chart)
    3. Precision-Recall tradeoff
    
    Args:
        metrics (dict): Metrics dictionary from evaluate_agent()
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Q-Learning Agent Evaluation Results', 
                 fontsize=16, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Subplot 1: Confusion Matrix
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    
    confusion = np.array([
        [metrics['true_negatives'], metrics['false_positives']],
        [metrics['false_negatives'], metrics['true_positives']]
    ])
    
    im = ax1.imshow(confusion, cmap='Blues', aspect='auto')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax1.text(j, i, confusion[i, j],
                          ha="center", va="center", 
                          color="white" if confusion[i, j] > confusion.max()/2 else "black",
                          fontsize=20, fontweight='bold')
    
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Predicted: Legit', 'Predicted: Fraud'])
    ax1.set_yticklabels(['Actual: Legit', 'Actual: Fraud'])
    ax1.set_title('Confusion Matrix', fontweight='bold')
    plt.colorbar(im, ax=ax1)
    
    # -------------------------------------------------------------------------
    # Subplot 2: Performance Metrics Bar Chart
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    
    metric_names = ['Detection\nRate', 'Precision', 'F1\nScore', 'Accuracy']
    metric_values = [
        metrics['detection_rate'],
        metrics['precision'],
        metrics['f1_score'],
        metrics['accuracy']
    ]
    
    bars = ax2.bar(metric_names, metric_values, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
    ax2.set_ylim([0, 1.0])
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Performance Metrics', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Subplot 3: Error Analysis
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    
    errors = {
        'False\nNegatives\n(Missed Fraud)': metrics['false_negatives'],
        'False\nPositives\n(False Alarms)': metrics['false_positives']
    }
    
    bars = ax3.bar(errors.keys(), errors.values(), color=['#e74c3c', '#f39c12'])
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Error Analysis', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Subplot 4: Class Distribution and Predictions
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    
    # Actual vs Predicted distribution
    actual_fraud = metrics['true_positives'] + metrics['false_negatives']
    actual_legit = metrics['true_negatives'] + metrics['false_positives']
    pred_fraud = metrics['true_positives'] + metrics['false_positives']
    pred_legit = metrics['true_negatives'] + metrics['false_negatives']
    
    x = np.arange(2)
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, [actual_legit, actual_fraud], 
                    width, label='Actual', color='#3498db', alpha=0.8)
    bars2 = ax4.bar(x + width/2, [pred_legit, pred_fraud], 
                    width, label='Predicted', color='#e74c3c', alpha=0.8)
    
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Actual vs Predicted Distribution', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Legitimate', 'Fraud'])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()


def plot_q_values(agent, states_to_plot=None):
    """
    Visualize learned Q-values for interpretability.
    
    Shows which states have highest/lowest values and how
    the agent's policy differs across states.
    
    Args:
        agent: Trained QLearningAgent
        states_to_plot (list): Specific states to visualize (None = all)
    """
    if not agent.q:
        print("No Q-values to plot (agent not trained)")
        return
    
    # Get all unique states
    states = list(set([k[0] for k in agent.q.keys()]))
    
    if states_to_plot:
        states = [s for s in states if s in states_to_plot]
    
    # Limit number of states for readability
    states = states[:20]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data
    state_labels = [f"{s[0]}\n{s[1]}\n{s[2]}" for s in states]
    approve_values = [agent.get_q_value(s, 0) for s in states]
    flag_values = [agent.get_q_value(s, 1) for s in states]
    
    x = np.arange(len(states))
    width = 0.35
    
    # Plot bars
    bars1 = ax.bar(x - width/2, approve_values, width, 
                   label='Approve (Action 0)', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, flag_values, width, 
                   label='Flag as Fraud (Action 1)', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('State (Amount, Time, Risk)', fontsize=12)
    ax.set_ylabel('Q-Value', fontsize=12)
    ax.set_title('Learned Q-Values by State', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(state_labels, fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()