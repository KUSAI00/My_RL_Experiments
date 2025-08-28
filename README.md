# My_RL_Experiments

This repository is a collection of Jupyter notebooks where I implement, dissect, and reflect on core RL algorithms—from the classics to the modern ones.

Whether you're a fellow student, a researcher, or just someone curious, I hope this repository offers clarity, and insight.

---

## Why This Exists

This repo is my way of:
- **Learning deeply** by building from scratch.
- **Documenting experiments** with clarity and reproducibility.
- **Sharing insights** with others who care about elegant design and practical impact.
- **Show companies** that i can really work and get accepted in the job.
---

## Notebooks:

- [GridWorld.ipynb](#gridworldipynb--tabular-rl-exploration)
- [DQN.ipynb](#dqnipynb--deep-q-network)
- [PPO.ipynb](#ppoipynb--proximal-policy-optimization)
- [A2C.ipynb](#a2cipynb--advantage-actor-critic)
- [REINFORCE.ipynb](#reinforceipynb--monte-carlo-policy-gradient)
- [DDPG.ipynb](#ddpgipynb--deep-deterministic-policy-gradient)
- [GAIL.ipynb](#gailipynb--generative-adversarial-imitation-learning)

---

## Notebook Descriptions

### `GridWorld.ipynb` – Tabular value iteration
I made two algorithims Value iteration and Q-value iteration

Value Iteration is an algorithm that computes the optimal value function V*(s) for each state s and, from that, the corresponding optimal policy π*.
Algorithm Steps
1. Initialize V(s) for all states s to arbitrary values.
2. Iteratively Update: Repeat the following update for all states s until the values converge: Vk+1(s)←max∑s′P(s′∣s,a)R(s,a,s′)+γVk(s′)]
3. Extract Optimal Policy: Once the values have converged to V∗(s), the optimal policy π∗(s) is deterministic and is found by taking the action that maximizes the future value from that state

Q-Value Iteration directly computes the optimal action-value function Q(s,a), which represents the value of taking action a in state s. This approach bypasses the need to calculate the state value function V(s) separately.
Algorithm Steps
1. Initialize Q(s,a) for all state-action pairs (s,a) to arbitrary values
2. Iteratively Update: Repeat the update rule above for all state-action pairs (s,a) until the Q-values converge.
3. Extract Optimal Policy: Once the Q-values have converged, the optimal policy is found directly by choosing the action with the highest Q-value for each state: π∗(s)=argmaxQ(s,a)
---






















### `DQN.ipynb` – Deep Q-Network  
Implements the classic DQN algorithm using experience replay and target networks.  
- Environment: CartPole-v1  
- Highlights:  
  - ε-greedy exploration  
  - Replay buffer implementation  
  - Target network synchronization  
- Visualization: Episode rewards, Q-value evolution  
- Notes: Includes hyperparameter tuning and training stability tips.

---

### `PPO.ipynb` – Proximal Policy Optimization  
A robust policy gradient method with clipped objective and adaptive updates.  
- Environment: LunarLander-v2  
- Highlights:  
  - Actor-Critic architecture  
  - Clipped surrogate loss  
  - Mini-batch training with advantage estimation  
- Visualization: Policy entropy, reward curves  
- Notes: Compared with vanilla policy gradients for stability.

---

### `A2C.ipynb` – Advantage Actor-Critic  
A synchronous version of A3C with shared networks for policy and value estimation.  
- Environment: MountainCarContinuous-v0  
- Highlights:  
  - Advantage calculation  
  - Shared neural network backbone  
  - On-policy updates  
- Visualization: Learning curves, value function heatmaps  
- Notes: Includes discussion on bias-variance tradeoff.

---

### `REINFORCE.ipynb` – Monte Carlo Policy Gradient  
Implements the simplest policy gradient method using full episode returns.  
- Environment: CartPole-v1  
- Highlights:  
  - Baseline subtraction  
  - Episode-based updates  
  - High variance handling  
- Visualization: Reward trajectory per episode  
- Notes: Great for understanding the fundamentals of policy gradients.

---

### `DDPG.ipynb` – Deep Deterministic Policy Gradient  
A continuous control algorithm combining actor-critic with target networks.  
- Environment: Pendulum-v0  
- Highlights:  
  - Deterministic policy  
  - Soft target updates  
  - Ornstein-Uhlenbeck noise for exploration  
- Visualization: Action trajectories, reward curves  
- Notes: Includes comparison with discrete-action methods.

---

### `GAIL.ipynb` – Generative Adversarial Imitation Learning  
Trains agents to mimic expert behavior using adversarial training.  
- Environment: Custom GridWorld  
- Highlights:  
  - Discriminator vs. policy training loop  
  - Expert trajectory loading  
  - Reward shaping from discriminator output  
- Visualization: Policy convergence, discriminator loss  
- Notes: Bridges RL and imitation learning with GAN-style training.

---


## 🛠️ Tools & Libraries

- Python 3.8^
- TensorFlow / PyTorch (varies by notebook)
- OpenAI Gym
- NumPy, Matplotlib, Seaborn
- Jupyter Notebook


