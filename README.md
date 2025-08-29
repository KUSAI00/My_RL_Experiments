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

- [GridWorld.ipynb](#gridworldipynb--tabular-value-iteration)
- [DQN.ipynb](#dqnipynb--deep-q-network)
- [REINFORCE.ipynb](#reinforceipynb--monte-carlo-policy-gradient)
- [A2C.ipynb](#a2cipynb--advantage-actor-critic)
- [PPO.ipynb](#ppoipynb--proximal-policy-optimization)
- [DDPG.ipynb](#ddpgipynb--deep-deterministic-policy-gradient)

---

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

The foundation of Q-Learning is the Bellman equation, which for an optimal policy can be written as:

Q*(s, a) = E[r + γ max_{a'} Q*(s', a') | s, a]

Algorithm Steps:
1. Initilize:
  - Initialize the main Q-network with random weights.
  - Initialize the target Q-network with the same weights as the main one.
2. loop over episodes:
  - Reset the environment to an initial state s.
  - For each step in the episode:
    - Select an action a using Epsilon-Greedy Policy
    - Execute action a in the environment, observe reward r, and the next state s′.
    - Store the experience (s,a,r,s′) in the replay buffer.
    - Update the current state.
    - Sample a random mini-batch of experiences from the replay buffer.
    - For each experience in the mini-batch
      - Calculate the target Q-value
      - Calculate the predicted Q-value for the taken action
      - Compute the loss
      - Perform gradient descent
  - epsilon decay
  - update target Q-network

---
### `REINFORCE.ipynb` – Monte Carlo Policy Gradient  
REINFORCE is a classic policy gradient method that learns a parameterized policy directly without using a value function. It uses the Monte Carlo method to estimate the policy gradient and updates the policy parameters to maximize expected rewards.
The idea is to directly optimize the policy parameters θ to maximize the expected return:

J(θ) = E[R_t | π_θ]

Where:

- J(θ) is the objective function (expected return)
- π_θ is the policy parameterized by θ
- R_t is the return (cumulative discounted reward)

The policy gradient theorem shows that:

∇_θ J(θ) = E[∇_θ log π_θ(a_t|s_t) * G_t]

Where:

- ∇_θ log π_θ(a_t|s_t) is the gradient of the log probability of action a_t given state s_t
- G_t is the return from time step t

The REINFORCE algorithm uses Monte Carlo estimation to approximate this gradient:

θ ← θ + α * ∇_θ log π_θ(a_t|s_t) * G_t

Where:

- α is the learning rate
- G_t is the actual return experienced from time t

The return G_t is calculated as the sum of discounted future rewards:

G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... = Σ_{k=0}^{T-t-1} γ^k R_{t+k+1}

Where:

- γ (gamma) is the discount factor (0 ≤ γ ≤ 1)
- T is the terminal time step

To reduce variance, we often normalize the returns

Algorithm Steps

1. Initialize policy network π_θ with random parameters θ
2. For each episode:

  - Generate an episode following π_θ: s_0, a_0, r_1, s_1, a_1, r_2, ..., s_{T-1}, a_{T-1}, r_T
  - For each time step t in the episode:
    - Calculate return G_t = Σ_{k=t+1}^T γ^{k-t-1} r_k
    - Calculate policy gradient: ∇θ J(θ) ≈ (1/T) Σ{t=0}^{T-1} ∇_θ log π_θ(a_t|s_t) G_t
    - Update parameters: θ ← θ + α ∇_θ J(θ)

---

### `A2C.ipynb` – Advantage Actor-Critic  
A2C is a policy gradient method that combines the benefits of both actor-critic methods and advantage estimation. Unlike REINFORCE, which only learns a policy, A2C learns both a policy (actor) and a value function (critic) simultaneously, leading to more stable and efficient learning.
The policy gradient for maximizing expected return J(θ) is:

∇_θ J(θ) = E[∇_θ log π_θ(a_t|s_t) * A^π(s_t, a_t)]

Where A^π(s_t, a_t) is the advantage function.
The advantage function measures how much better an action is compared to the average:

A^π(s_t, a_t) = Q^π(s_t, a_t) - V^π(s_t)

In A2C, we approximate this using the Temporal Difference (TD) error:

A(s_t, a_t) ≈ r_t + γV(s_{t+1}) - V(s_t)

For episodic tasks (like LunarLander), we use Monte Carlo returns:

A(s_t, a_t) = G_t - V(s_t)

Where G_t is the discounted return from time step t.

Actor Loss (Policy Loss)

The actor loss aims to maximize expected returns:

L_actor = -E[log π_θ(a_t|s_t) * A(s_t, a_t)]

We use the negative because optimizers minimize loss, but we want to maximize returns.
Critic Loss (Value Loss)
The critic loss is the mean squared error between predicted and actual returns:

L_critic = E[(G_t - V_φ(s_t))²]

Combined Loss

The total loss combines both components:

L_total = L_actor + λ * L_critic

Where λ is a weighting factor (typically 0.5).

Algorithm Steps

1. Initialize actor-critic network with parameters θ (actor) and φ (critic)
2. For each episode:
  - Reset environment and get initial state s_0
  -For each time step t:
    - Get action probabilities π_θ(·|s_t) and value V_φ(s_t) from network
    - Sample action a_t from π_θ(·|s_t)
    - Execute action, receive reward r_t and next state s_{t+1}
    - Store (s_t, a_t, r_t, done_t)
  - At episode end:
    - Calculate returns G_t for all time steps
    - Calculate advantages A_t = G_t - V_φ(s_t)
    - Update actor: θ ← θ + α_actor * ∇_θ Σ_t log π_θ(a_t|s_t) * A_t
    - Update critic: φ ← φ - α_critic * ∇_φ Σ_t (G_t - V_φ(s_t))²

---

### `PPO.ipynb` – Proximal Policy Optimization  
PPO is a state-of-the-art policy gradient method that addresses the sample efficiency and training stability issues of earlier methods like REINFORCE and vanilla policy gradients. It introduces a clipped surrogate objective that prevents destructively large policy updates while maintaining the benefits of policy gradient methods.
Traditional policy gradient methods can suffer from:

- Large policy updates that destabilize training
- Sample inefficiency requiring new data after each update
- Difficulty in choosing appropriate step sizes

PPO solves these issues by constraining policy updates to stay within a "trust region."
The key innovation of PPO is the clipped surrogate objective function:

L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]

Where:
- r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) is the probability ratio
- A_t is the advantage at time step t
- ε (epsilon) is the clipping parameter (typically 0.1 to 0.3)
- clip(x, a, b) clamps x to the range [a, b]

Advantage Function calculated using Monte Carlo returns for episodic tasks:

A_t = G_t - V(s_t)

G_t = Σ_{k=0}^{T-t-1} γ^k r_{t+k+1}

The clipping ensures that:

- If A_t > 0 (good action): ratio is clipped to [1, 1+ε]
- If A_t < 0 (bad action): ratio is clipped to [1-ε, 1]

This prevents the policy from changing too drastically in a single update.

The critic is trained using mean squared error:

L^VF(φ) = E[(V_φ(s_t) - G_t)²]

The total loss function is:

L(θ,φ) = E[L^CLIP(θ) + c₁L^VF(φ) - c₂H(π_θ)]

Where:
- c₁ is the value function coefficient (typically 0.5)
- c₂ is the entropy coefficient (optional, for exploration)
- H(π_θ) is the entropy of the policy

Algorithm Steps
1. Initialize actor π_θ and critic V_φ networks
2. For each iteration:
  - Collect experience using current policy π_θ_old
  - For K epochs:
    - For each mini-batch:
      - Calculate advantages A_t = G_t - V_φ(s_t)
      - Calculate probability ratio r_t(θ)
      - Compute clipped surrogate loss L^CLIP(θ)
      - Compute value function loss L^VF(φ)
      - Update networks using gradients
  - Set θ_old ← θ (update old policy for next iteration)

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



## 🛠️ Tools & Libraries

- Python 3.8^
- TensorFlow / PyTorch (varies by notebook)
- OpenAI Gym
- NumPy, Matplotlib, Seaborn
- Jupyter Notebook


