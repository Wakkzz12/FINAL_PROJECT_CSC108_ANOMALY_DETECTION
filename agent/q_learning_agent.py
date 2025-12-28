import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions, alpha, gamma, epsilon):
        self.q = defaultdict(float)
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.actions, key=lambda a: self.q[(state, a)])

    def update(self, state, action, reward, next_state):
        best_next = max(self.actions, key=lambda a: self.q[(next_state, a)]) if next_state else 0
        self.q[(state, action)] += self.alpha * (
            reward + self.gamma * best_next - self.q[(state, action)]
        )
