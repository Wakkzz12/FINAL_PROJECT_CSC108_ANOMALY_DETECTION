def get_reward(action, label):
    """
    Fine-tuned reward function for F1 > 0.80
    """
    if action == 0 and label == 0:
        return 5      # ← Increased: Reward legitimate approvals more
    if action == 1 and label == 1:
        return 50     
    if action == 0 and label == 1:
        return -80    
    if action == 1 and label == 0:
        return -25    # ← Increased: Penalize false alarms more