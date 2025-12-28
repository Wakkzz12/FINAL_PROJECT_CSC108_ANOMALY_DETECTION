from reward.reward_function import get_reward

def train(agent, env, episodes, epsilon_decay, min_epsilon):
    rewards_history = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            label, next_state, done = env.step(action)

            reward = get_reward(action, label)
            agent.update(state, action, reward, next_state)

            total_reward += reward
            state = next_state

        agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)
        rewards_history.append(total_reward)

    return rewards_history
