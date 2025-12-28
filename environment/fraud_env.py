from utils.discretizer import get_state

class FraudEnvironment:
    def __init__(self, data, bins):
        self.data = data.reset_index(drop=True)
        self.bins = bins
        self.index = 0

    def reset(self):
        self.index = 0
        return get_state(self.data.iloc[self.index], self.bins)

    def step(self, action):
        row = self.data.iloc[self.index]
        label = row["Class"]

        self.index += 1
        done = self.index >= len(self.data) - 1

        next_state = get_state(self.data.iloc[self.index], self.bins) if not done else None
        return label, next_state, done
