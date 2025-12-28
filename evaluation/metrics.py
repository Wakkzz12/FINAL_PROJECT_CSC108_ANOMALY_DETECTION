# =============================================================================
# EVALUATION METRICS FOR FRAUD DETECTION
# =============================================================================
# Standard classification metrics for binary classification tasks
# Particularly important for imbalanced datasets like fraud detection
# =============================================================================

def detection_rate(tp, fn):
    """
    Calculate Detection Rate (Recall / True Positive Rate / Sensitivity).
    
    Detection Rate = TP / (TP + FN)
    
    Measures: What proportion of actual frauds did we catch?
    - High detection rate = catching most frauds (good!)
    - Critical metric for fraud detection (missing fraud is costly)
    
    Args:
        tp (int): True Positives (correctly flagged frauds)
        fn (int): False Negatives (missed frauds)
        
    Returns:
        float: Detection rate [0.0 to 1.0]
    """
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def false_positive_rate(fp, tn):
    """
    Calculate False Positive Rate (Type I Error Rate).
    
    FPR = FP / (FP + TN)
    
    Measures: What proportion of legitimate transactions did we incorrectly flag?
    - Low FPR = fewer false alarms (good!)
    - Too many false alarms annoy customers and waste resources
    
    Args:
        fp (int): False Positives (legitimate flagged as fraud)
        tn (int): True Negatives (correctly approved legitimate)
        
    Returns:
        float: False positive rate [0.0 to 1.0]
    """
    return fp / (fp + tn) if (fp + tn) > 0 else 0


def precision(tp, fp):
    """
    Calculate Precision (Positive Predictive Value).
    
    Precision = TP / (TP + FP)
    
    Measures: When we flag a transaction, how often are we correct?
    - High precision = most flags are real frauds (good!)
    - Important when acting on flags is expensive (investigations, blocks)
    
    Args:
        tp (int): True Positives
        fp (int): False Positives
        
    Returns:
        float: Precision [0.0 to 1.0]
    """
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def accuracy(tp, tn, fp, fn):
    """
    Calculate Overall Accuracy.
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Measures: What proportion of all predictions were correct?
    - NOTE: Misleading for imbalanced datasets!
    - Can have 99.8% accuracy by just approving everything (useless for fraud)
    
    Args:
        tp (int): True Positives
        tn (int): True Negatives
        fp (int): False Positives
        fn (int): False Negatives
        
    Returns:
        float: Accuracy [0.0 to 1.0]
    """
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0


def f1_score(tp, fp, fn):
    """
    Calculate F1 Score (Harmonic Mean of Precision and Recall).
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Measures: Balanced measure between precision and recall
    - Better metric than accuracy for imbalanced datasets
    - Reaches best value at 1 (perfect) and worst at 0
    
    Args:
        tp (int): True Positives
        fp (int): False Positives
        fn (int): False Negatives
        
    Returns:
        float: F1 score [0.0 to 1.0]
    """
    prec = precision(tp, fp)
    rec = detection_rate(tp, fn)
    
    if prec + rec == 0:
        return 0
    
    return 2 * (prec * rec) / (prec + rec)


def evaluate_agent(agent, env, data):
    """
    Comprehensive evaluation of trained Q-Learning agent.
    
    Runs agent through all transactions in dataset using learned policy
    (greedy action selection, no exploration) and computes all metrics.
    
    Time Complexity: O(N * |A|) where N = transactions, |A| = actions
    Space Complexity: O(1) - only tracking counters
    
    Args:
        agent: Trained QLearningAgent
        env: FraudEnvironment with test data
        data (pd.DataFrame): Test dataset
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    # Initialize confusion matrix counters
    tp = 0  # True Positives: Correctly flagged fraud
    tn = 0  # True Negatives: Correctly approved legitimate
    fp = 0  # False Positives: Flagged legitimate as fraud
    fn = 0  # False Negatives: Approved fraud (missed)
    
    # Reset environment and temporarily disable exploration
    state = env.reset()
    old_epsilon = agent.epsilon
    agent.epsilon = 0  # Pure exploitation for evaluation
    
    done = False
    
    # Iterate through all transactions
    # Time Complexity: O(N) iterations
    while not done:
        # Get agent's action using learned policy (greedy)
        action = agent.get_greedy_action(state)  # O(|A|)
        
        # Execute action and get true label
        label, next_state, done = env.step(action)
        
        # Update confusion matrix
        # action: 0 = approve, 1 = flag
        # label: 0 = legitimate, 1 = fraud
        if action == 0 and label == 0:
            tn += 1  # Correct: Approved legitimate
        elif action == 1 and label == 1:
            tp += 1  # Correct: Flagged fraud
        elif action == 0 and label == 1:
            fn += 1  # Error: Missed fraud
        elif action == 1 and label == 0:
            fp += 1  # Error: False alarm
        
        state = next_state
    
    # Restore original epsilon
    agent.epsilon = old_epsilon
    
    # Calculate all metrics
    metrics = {
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'detection_rate': detection_rate(tp, fn),
        'false_positive_rate': false_positive_rate(fp, tn),
        'precision': precision(tp, fp),
        'accuracy': accuracy(tp, tn, fp, fn),
        'f1_score': f1_score(tp, fp, fn)
    }
    
    return metrics


def print_evaluation_results(metrics):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics (dict): Metrics dictionary from evaluate_agent()
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\nCONFUSION MATRIX:")
    print(f"  True Positives (Caught Fraud):      {metrics['true_positives']:5d}")
    print(f"  True Negatives (Approved Legit):    {metrics['true_negatives']:5d}")
    print(f"  False Positives (False Alarms):     {metrics['false_positives']:5d}")
    print(f"  False Negatives (Missed Fraud):     {metrics['false_negatives']:5d}")
    
    print("\nPERFORMANCE METRICS:")
    print(f"  Detection Rate (Recall):     {metrics['detection_rate']:.4f} ({metrics['detection_rate']*100:.2f}%)")
    print(f"  Precision:                   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  F1 Score:                    {metrics['f1_score']:.4f}")
    print(f"  Accuracy:                    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  False Positive Rate:         {metrics['false_positive_rate']:.4f} ({metrics['false_positive_rate']*100:.2f}%)")
    
    print("\nINTERPRETATION:")
    if metrics['detection_rate'] > 0.8:
        print("  ✓ Excellent fraud detection rate!")
    elif metrics['detection_rate'] > 0.6:
        print("  ✓ Good fraud detection rate")
    else:
        print("  ✗ Low fraud detection - consider retraining")
    
    if metrics['false_positive_rate'] < 0.1:
        print("  ✓ Low false alarm rate")
    elif metrics['false_positive_rate'] < 0.2:
        print("  ~ Moderate false alarm rate")
    else:
        print("  ✗ High false alarm rate - may annoy customers")
    
    print("="*60 + "\n")