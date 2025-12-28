from config.config import *
from data.loader import load_data
from environment.fraud_env import FraudEnvironment
from agent.q_learning_agent import QLearningAgent
from training.trainer import train
from visualization.plots import plot_rewards

data = load_data("creditcard.csv")

env = FraudEnvironment(data, AMOUNT_BINS)
agent = QLearningAgent(actions=[0, 1], alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)

rewards = train(agent, env, EPISODES, EPSILON_DECAY, MIN_EPSILON)
plot_rewards(rewards)
