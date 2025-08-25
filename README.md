# 🧠 My_RL_Experiments

This repository is a collection of Jupyter notebooks where I implement, dissect, and reflect on core RL algorithms—from the classics to the more nuanced.

Whether you're a fellow student, a researcher, or just someone curious, I hope this repository offers clarity, and insight.

---

## 🚀 Why This Exists

As a master's student in Artificial Intelligence with a background in Civil Engineering. Reinforcement Learning sits at the intersection of control, decision-making, and curiosity—making it the perfect domain to explore both theory and practice.

This repo is my way of:
- **Learning deeply** by building from scratch.
- **Documenting experiments** with clarity and reproducibility.
- **Sharing insights** with others who care about elegant design and practical impact.
- **Show companies** that i can really work and get accepted in the job.
---

## 📚 What You'll Find Here

## 📓 Notebooks Overview

- [DQN.ipynb](#dqnipynb--deep-q-network)
- [PPO.ipynb](#ppoipynb--proximal-policy-optimization)
- [A2C.ipynb](#a2cipynb--advantage-actor-critic)
- [REINFORCE.ipynb](#reinforceipynb--monte-carlo-policy-gradient)
- [DDPG.ipynb](#ddpgipynb--deep-deterministic-policy-gradient)
- [GAIL.ipynb](#gailipynb--generative-adversarial-imitation-learning)
- [GridWorld.ipynb](#gridworldipynb--tabular-rl-exploration)

---

## 📘 Notebook Descriptions

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

### `GridWorld.ipynb` – Tabular RL Exploration  
A visual and intuitive intro to RL using a simple grid-based environment.  
- Environment: Custom GridWorld  
- Highlights:  
  - Q-learning and SARSA  
  - Value iteration and policy iteration  
  - Heatmap visualization of value functions  
- Visualization: Grid overlays, policy arrows  
- Notes: Ideal for beginners and algorithm debugging.
---

## 🛠️ Tools & Libraries

- Python 3.8^
- TensorFlow / PyTorch (varies by notebook)
- OpenAI Gym
- NumPy, Matplotlib, Seaborn
- Jupyter Notebook


