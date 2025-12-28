from reward.reward_function import get_reward

def train(agent, env, episodes, epsilon_decay, min_epsilon):
    rewards_history = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            label, next_state, done = env.step(action)

            reward = get_reward(action, label)
            
            # Handle terminal state properly
            agent.update(state, action, reward, next_state)

            total_reward += reward
            state = next_state if next_state is not None else state

        # Decay epsilon
        agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    return rewards_history
