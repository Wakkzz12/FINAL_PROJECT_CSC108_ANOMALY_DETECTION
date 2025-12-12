# Q LEARNING ALGORITHM ON ANOMALY DETECTION
Q-Learning is an off-policy (can update the estimated value functions using hypothetical actions, those which have not actually been tried) algorithm for temporal difference learning ( method to estimate value functions). It can be proven that given sufficient training, the Q-learning converges with probability 1 to a close approximation of the action-value function for an arbitrary target policy. Q-Learning learns the optimal policy even when actions are selected according to a more exploratory or even random policy. Q-learning can be implemented as follows:


```
Initialize Q(s,a) arbitrarily
Repeat (for each generation):
	Initialize state s
	While (s is not a terminal state):		
		Choose a from s using policy derived from Q
		Take action a, observe r, s'
		Q(s,a) += alpha * (r + gamma * max,Q(s') - Q(s,a))
		s = s'
```

### WHERE:
- **s:** is the previous state
- **a:** is the previous action
- **Q():** is the Q-learning algorithm
- **s':** is the current state
- **alpha:** is the the learning rate, set generally between 0 and 1. Setting it to 0 means that the Q-values are never updated, thereby nothing is learned. Setting alpha to a high value such as 0.9 means that learning can occur quickly.
- **gamma:** is the discount factor, also set between 0 and 1. This models the fact that future rewards are worth less than immediate rewards.
- **max,:** is the the maximum reward that is attainable in the state following the current one (the reward for taking the optimal action thereafter).  

### THE ALGORITHM CAN BE INTERPRETED AS:

- **Initialize the Q-values table, Q(s, a)**
- **Observe the current state, s.**
- **Choose an action, a, for that state based on the selection policy.**
- **Take the action, and observe the reward, r, as well as the new state, s'.**
- **Update the Q-value for the state using the observed reward and the maximum reward possible for the next state.**
- **Set the state to the new state, and repeat the process until a terminal state is reached.**


### SETUP:
Install Python 3

