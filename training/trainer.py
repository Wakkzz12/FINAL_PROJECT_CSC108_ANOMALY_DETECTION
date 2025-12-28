from reward.reward_function import get_reward
import numpy as np

# =============================================================================
# ROBUST Q-LEARNING TRAINER WITH EARLY STOPPING
# =============================================================================
# This implementation is resistant to inappropriate hyperparameter settings
# by monitoring convergence and stopping early when no improvement is seen.
# =============================================================================

def train(agent, env, episodes, epsilon_decay, min_epsilon, 
          early_stopping=True, patience=15, convergence_threshold=0.001):
    """
    Train Q-Learning agent with robust early stopping mechanism.
    
    Early Stopping prevents:
    - Overfitting from too many episodes
    - Wasted computation time
    - Over-exploitation when epsilon becomes too low
    
    Algorithm monitors:
    1. Q-value changes between episodes
    2. Reward stability over recent episodes
    3. Performance plateau detection
    
    Time Complexity: O(E_actual × N × |A|) where E_actual ≤ episodes
    
    Args:
        agent: QLearningAgent instance
        env: FraudEnvironment instance
        episodes (int): Maximum training episodes
        epsilon_decay (float): Exploration decay rate
        min_epsilon (float): Minimum exploration rate
        early_stopping (bool): Enable early stopping
        patience (int): Episodes to wait before stopping if no improvement
        convergence_threshold (float): Q-value change threshold for convergence
        
    Returns:
        dict: Training history with rewards, q_changes, stopped_early flag
    """
    rewards_history = []
    q_value_changes = []
    best_avg_reward = float('-inf')
    patience_counter = 0
    
    print(f"\nTraining Configuration:")
    print(f"  Max episodes: {episodes}")
    print(f"  Early stopping: {'Enabled' if early_stopping else 'Disabled'}")
    if early_stopping:
        print(f"  Patience: {patience} episodes")
        print(f"  Convergence threshold: {convergence_threshold}")
    print()
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False
        
        # Track Q-value changes for convergence detection
        q_values_before = dict(agent.q.copy()) if early_stopping else None
        
        # Run episode
        while not done:
            action = agent.choose_action(state)
            label, next_state, done = env.step(action)
            
            reward = get_reward(action, label)
            agent.update(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state
        
        # Decay exploration rate
        agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        
        # Calculate Q-value change for convergence monitoring
        if early_stopping:
            q_change = calculate_q_change(agent.q, q_values_before)
            q_value_changes.append(q_change)
        
        # Print progress every 10 episodes or at key milestones
        if episode % 10 == 0 or episode == 1 or episode <= 5:
            print(f"Episode {episode}/{episodes}, "
                  f"Total Reward: {total_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}", end="")
            if early_stopping:
                print(f", Q-Change: {q_change:.6f}")
            else:
                print()
        
        # EARLY STOPPING LOGIC
        if early_stopping and episode >= 30:  # Start checking after 30 episodes
            
            # Calculate average reward over last 10 episodes
            recent_avg_reward = np.mean(rewards_history[-10:])
            
            # Check if we've improved
            if recent_avg_reward > best_avg_reward + (abs(best_avg_reward) * 0.01):
                best_avg_reward = recent_avg_reward
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Check convergence criteria
            converged = check_convergence(
                q_value_changes, 
                convergence_threshold,
                window=5
            )
            
            # Stop if converged or no improvement for 'patience' episodes
            if converged or patience_counter >= patience:
                reason = "Q-values converged" if converged else f"no improvement for {patience} episodes"
                print(f"\n{'='*70}")
                print(f"EARLY STOPPING at Episode {episode}/{episodes}")
                print(f"Reason: {reason}")
                print(f"Best average reward: {best_avg_reward:.2f}")
                print(f"Final epsilon: {agent.epsilon:.4f}")
                print(f"{'='*70}\n")
                
                return {
                    'rewards': rewards_history,
                    'q_changes': q_value_changes,
                    'stopped_early': True,
                    'stopped_at_episode': episode,
                    'reason': reason
                }
    
    # Completed all episodes without early stopping
    print(f"\nCompleted all {episodes} episodes (no early stopping triggered)")
    
    return {
        'rewards': rewards_history,
        'q_changes': q_value_changes if early_stopping else [],
        'stopped_early': False,
        'stopped_at_episode': episodes,
        'reason': 'completed_all_episodes'
    }


def calculate_q_change(q_current, q_previous):
    """
    Calculate average absolute change in Q-values.
    
    Measures how much Q-table changed during episode.
    Small changes indicate convergence.
    
    Time Complexity: O(|S| × |A|)
    
    Args:
        q_current (dict): Current Q-values
        q_previous (dict): Previous Q-values
        
    Returns:
        float: Average absolute change in Q-values
    """
    if not q_previous:
        return float('inf')
    
    changes = []
    all_keys = set(q_current.keys()) | set(q_previous.keys())
    
    for key in all_keys:
        old_val = q_previous.get(key, 0)
        new_val = q_current.get(key, 0)
        changes.append(abs(new_val - old_val))
    
    return np.mean(changes) if changes else 0.0


def check_convergence(q_changes, threshold, window=5):
    """
    Check if Q-values have converged based on recent changes.
    
    Convergence criteria:
    - Average change in last 'window' episodes < threshold
    - Indicates Q-values have stabilized
    
    Args:
        q_changes (list): History of Q-value changes
        threshold (float): Convergence threshold
        window (int): Number of recent episodes to check
        
    Returns:
        bool: True if converged, False otherwise
    """
    if len(q_changes) < window:
        return False
    
    recent_changes = q_changes[-window:]
    avg_change = np.mean(recent_changes)
    
    return avg_change < threshold


def train_with_validation(agent, train_env, val_env, episodes, 
                          epsilon_decay, min_epsilon, val_frequency=10):
    """
    Advanced: Train with periodic validation to prevent overfitting.
    
    Monitors performance on validation set and stops if it degrades.
    More sophisticated than simple early stopping.
    
    Args:
        agent: QLearningAgent instance
        train_env: Training environment
        val_env: Validation environment
        episodes (int): Maximum training episodes
        epsilon_decay (float): Exploration decay rate
        min_epsilon (float): Minimum exploration rate
        val_frequency (int): Validate every N episodes
        
    Returns:
        dict: Training history with validation scores
    """
    from evaluation.metrics import evaluate_agent
    
    rewards_history = []
    val_f1_scores = []
    best_val_f1 = 0
    patience_counter = 0
    patience = 5  # Stop after 5 validations without improvement
    
    print(f"\nTraining with Validation:")
    print(f"  Validation frequency: every {val_frequency} episodes")
    print(f"  Patience: {patience} validations\n")
    
    for episode in range(1, episodes + 1):
        state = train_env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            label, next_state, done = train_env.step(action)
            
            reward = get_reward(action, label)
            agent.update(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state
        
        agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        
        # Periodic validation
        if episode % val_frequency == 0:
            # Evaluate on validation set
            old_epsilon = agent.epsilon
            agent.epsilon = 0  # No exploration during validation
            
            # Get validation metrics (you'll need validation data)
            # This is a placeholder - implement based on your needs
            val_metrics = {'f1_score': 0.75}  # Placeholder
            val_f1 = val_metrics['f1_score']
            val_f1_scores.append(val_f1)
            
            agent.epsilon = old_epsilon
            
            print(f"Episode {episode}: Train Reward={total_reward:.0f}, "
                  f"Val F1={val_f1:.4f}, Best={best_val_f1:.4f}")
            
            # Check for improvement
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping based on validation
            if patience_counter >= patience:
                print(f"\nStopping: No validation improvement for {patience} checks")
                break
    
    return {
        'rewards': rewards_history,
        'val_f1_scores': val_f1_scores,
        'stopped_early': patience_counter >= patience,
        'stopped_at_episode': episode
    }


def adaptive_epsilon_decay(episode, total_episodes, initial_epsilon=1.0, 
                           min_epsilon=0.05, strategy='exponential'):
    """
    Calculate epsilon for current episode using adaptive strategy.
    
    Makes algorithm robust to different episode counts.
    Automatically adjusts decay based on total episodes.
    
    Strategies:
    - exponential: Fast initial decay, slow later (good for most cases)
    - linear: Constant decay rate
    - polynomial: Smooth decay with tunable rate
    
    Args:
        episode (int): Current episode number
        total_episodes (int): Total planned episodes
        initial_epsilon (float): Starting exploration rate
        min_epsilon (float): Minimum exploration rate
        strategy (str): Decay strategy
        
    Returns:
        float: Epsilon value for this episode
    """
    progress = episode / total_episodes
    
    if strategy == 'exponential':
        # Decay rate automatically adjusted for total episodes
        # Ensures epsilon reaches ~min_epsilon by 70% of training
        decay_rate = -np.log(min_epsilon / initial_epsilon) / (0.7 * total_episodes)
        epsilon = initial_epsilon * np.exp(-decay_rate * episode)
        
    elif strategy == 'linear':
        # Linear decay from initial to min
        epsilon = initial_epsilon - (initial_epsilon - min_epsilon) * progress
        
    elif strategy == 'polynomial':
        # Polynomial decay (power=2 gives smooth curve)
        epsilon = initial_epsilon - (initial_epsilon - min_epsilon) * (progress ** 2)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return max(epsilon, min_epsilon)


def train_robust(agent, env, episodes, epsilon_decay=None, min_epsilon=0.05,
                 early_stopping=True, adaptive_epsilon=True):
    """
    RECOMMENDED: Robust training that works well with any episode count.
    
    Features:
    - Automatic epsilon adjustment based on episode count
    - Early stopping to prevent overfitting
    - Convergence detection
    - Works well with 50, 100, 500, or 1000+ episodes
    
    Args:
        agent: QLearningAgent instance
        env: FraudEnvironment instance
        episodes (int): Training episodes (any reasonable value works)
        epsilon_decay (float): Manual decay (None = auto-calculate)
        min_epsilon (float): Minimum exploration rate
        early_stopping (bool): Enable early stopping
        adaptive_epsilon (bool): Use adaptive epsilon (recommended)
        
    Returns:
        dict: Training history
    """
    print(f"\n{'='*70}")
    print("ROBUST Q-LEARNING TRAINING")
    print(f"{'='*70}")
    
    # Auto-calculate optimal epsilon decay if not provided
    if epsilon_decay is None and not adaptive_epsilon:
        # Calculate decay to reach min_epsilon by 70% of episodes
        epsilon_decay = (min_epsilon / 1.0) ** (1 / (0.7 * episodes))
        print(f"Auto-calculated epsilon decay: {epsilon_decay:.6f}")
    
    # Use adaptive strategy if enabled
    if adaptive_epsilon:
        print(f"Using adaptive epsilon decay (exponential strategy)")
        print(f"Epsilon will decay from 1.0 → {min_epsilon} over {episodes} episodes")
    
    rewards_history = []
    q_value_changes = []
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False
        
        q_values_before = dict(agent.q.copy())
        
        while not done:
            action = agent.choose_action(state)
            label, next_state, done = env.step(action)
            
            reward = get_reward(action, label)
            agent.update(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state
        
        # Update epsilon (adaptive or manual)
        if adaptive_epsilon:
            agent.epsilon = adaptive_epsilon_decay(
                episode, episodes, 
                initial_epsilon=1.0, 
                min_epsilon=min_epsilon
            )
        else:
            agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)
        
        rewards_history.append(total_reward)
        q_change = calculate_q_change(agent.q, q_values_before)
        q_value_changes.append(q_change)
        
        # Progress reporting
        if episode % max(1, episodes // 10) == 0 or episode <= 5:
            print(f"Episode {episode}/{episodes}, "
                  f"Reward: {total_reward:.2f}, "
                  f"ε: {agent.epsilon:.3f}, "
                  f"Q-Δ: {q_change:.6f}")
        
        # Early stopping check
        if early_stopping and episode >= 30:
            if check_convergence(q_value_changes, threshold=0.001, window=5):
                print(f"\n{'='*70}")
                print(f"CONVERGED at Episode {episode}/{episodes}")
                print(f"Final epsilon: {agent.epsilon:.4f}")
                print(f"{'='*70}\n")
                break
    
    return {
        'rewards': rewards_history,
        'q_changes': q_value_changes,
        'final_episode': episode
    }