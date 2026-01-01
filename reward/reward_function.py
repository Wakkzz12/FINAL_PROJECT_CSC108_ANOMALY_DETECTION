def get_reward(action, label):
    """
    Fine-tuned reward function for F1 > 0.80
    """
    if action == 0 and label == 0:
        return 15      # Correct: Approved legitimate
    if action == 1 and label == 1:
        return 50     # Correct: Flagged fraud
    if action == 0 and label == 1:
        return -60    # Error: Missed fraud
    if action == 1 and label == 0:
        return -40    # Error: Flagged legitimate