def get_reward(action, label):
    # action: 0 = approve, 1 = flag
    # label: 0 = legit, 1 = fraud

    if action == 0 and label == 0:
        return 5     # correct approval
    if action == 1 and label == 1:
        return 10    # correct fraud detection
    if action == 0 and label == 1:
        return -20   # fraud missed
    if action == 1 and label == 0:
        return -5    # false alarm
