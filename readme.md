# Credit Card Fraud Detection using Q-Learning (Anomaly Detection)

## Project Overview
This project implements a Credit Card Fraud Detection System using Q-Learning, a Reinforcement Learning (RL) algorithm. Fraud detection is modeled as an anomaly detection problem, where fraudulent transactions are considered rare and abnormal behaviors that deviate from normal transaction patterns.

The system learns a decision-making policy that determines whether a transaction should be approved or flagged as fraudulent based on reward-based feedback.

This project was developed for CSC108 – Algorithms and Complexity and follows the prescribed course learning outcomes and rubric.

---

## Objectives
- Apply Q-Learning (Reinforcement Learning) to a real-world application
- Model credit card fraud detection as anomaly detection
- Design a modular and maintainable Python system
- Analyze agent behavior using rewards, penalties, and learning curves

---

## Problem Description
Credit card fraud causes significant financial loss and security risks. Fraudulent transactions occur infrequently but differ significantly from legitimate spending behavior, making them anomalies in transaction data.

Instead of relying solely on supervised classification, this project uses reinforcement learning, where an agent interacts with transaction states, performs actions, and learns from reward feedback to minimize fraud-related loss over time.

---

## Algorithm Used: Q-Learning
Q-Learning is a model-free reinforcement learning algorithm that learns the optimal action-selection policy by estimating the expected rewards of actions taken in different states.

### Q-Learning Update Rule
Q(s, a) ← Q(s, a) + α [ r + γ max(Q(s′, a′)) − Q(s, a) ]


Where:
- s = current state
- a = action
- r = reward
- α = learning rate
- γ = discount factor

---

## Fraud Detection as Anomaly Detection
Fraud detection is treated as a specialized form of anomaly detection. Normal transactions follow consistent spending patterns, while fraudulent transactions deviate significantly from these patterns.

The system does not directly classify transactions using labels. Labels are used only to provide reward feedback, allowing the agent to learn anomalous behavior through interaction with the environment.

---

## State Representation
Each transaction is discretized into a state defined as:
State = (Transaction Amount Level, Time Pattern, Risk Level)


Possible values include:
- Amount Level: LOW, MEDIUM, HIGH
- Time Pattern: NORMAL, UNUSUAL
- Risk Level: LOW, HIGH

This discrete representation enables efficient Q-Learning while remaining interpretable.

---

## Actions
The agent can perform the following actions:
- 0 → Approve transaction
- 1 → Flag transaction as fraudulent

---

## Reward and Penalty Design
The reward system reflects real-world fraud detection costs:

| Outcome | Reward |
|------|------|
| Fraud correctly flagged | +15 |
| Legitimate transaction approved | +50 |
| Legitimate transaction flagged | -60 |
| Fraud approved | -40 |

This design encourages fraud prevention while minimizing unnecessary transaction blocking.

---
## Project Structure
FINAL_PROJECT_CSC108_ANOMALY_DETECTION/

├── data/

│ └── creditcard_subset.csv
│

├── environment/

│ └── environment.py
│

├── qlearning/

│ └── q_learning.py
│

├── reward/

│ └── reward_function.py
│

├── training/

│ └── trainer.py
│

├── utils/

│ └── discretizer.py
│

├── visualization/

│ └── plot_results.py
│

├── main.py

├── README.md

└── .gitignore

## Dataset
Due to GitHub file size limitations, the Credit Card Fraud dataset is not included in this repository.

Please download the dataset manually from: https://drive.google.com/file/d/1BpB5p_qOR6iJD0zSkZ7sE1M4KpJ_qdMH/view?usp=sharing

### Dataset Setup
Download `creditcard.csv` from gdgrive
Place the file inside the `data/` directory
Use the provided subset file for testing and development

### Install Dependencies
```
pip install pandas numpy matplotlib scikit-learn
```

```
run python main.py
```

```
on vscode (CRTL + ALT + N) to run
```




