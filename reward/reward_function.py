def get_reward(action, label):
    """
    Fine-tuned reward function for F1 > 0.80
    """
    if action == 0 and label == 0:
        return 15      # ← Increased: Reward legitimate approvals more
    if action == 1 and label == 1:
        return 50     
    if action == 0 and label == 1:
        return -60    
    if action == 1 and label == 0:
        return -40    # ← Increased: Penalize false alarms more