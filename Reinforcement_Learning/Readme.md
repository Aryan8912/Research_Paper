# Basics of Reinforcement Learning

 Reinforcement learning problems involve learning what to do, how to map situations to actions to maximize a numerical reward signal. They are closed-loop problems because the learning system's actions influence its later inputs.

 ## Elements of Reinforcement Learning:
 four main subelements of a reinforcement learning system: a `policy`, a `reward signal`, a `value function`, and, optionally, a `model of the environment`

 `Policy`:  A policy defines the learning agent's behavior at a given time.
 '

![image](https://github.com/user-attachments/assets/c2a0a0b1-ce6a-4a8c-b99c-b9298af45778)


## Bellman equation
![image](https://github.com/user-attachments/assets/db5f22fe-9143-442d-bf0f-6cd6300427c6)

## Markov Decision Processes

## Planning by Dynamic Programming

## Model-Free Prediction

## Model-Free Control

 Model-Free Reinforcement Learning
 
 Uses and Applications of Model-Free Control

 On and Off-Policy Learning
 
 Generalized Policy Iteration
 
 Generalized Policy Iteration with Monte-Carlo Evaluation
 
 Example of Greedy Action Selection
 
 Epsilon Greedy Exploration
 
 Epsilon Greedy Policy Improvement

 Monte-Carlo Policy Iteration
 
 Monte-Carlo Control
 
 GLIE (Greedy in the Limit with Infinite Exploration)
 
 GLIE Monte-Carlo Control
 
 Monte-Carlo Control in Blackjack

 MC vs TD Control
 Updating Action-Value Functions with Sarsa
 
 Sarsa Algorithm for On-Po licy Control
 
 Convergence of Sarsa
 
 Windy Gridworld Example
 
 n-Step Sarsa

 Forward View Sarsa(lambda)
 
 Backward View Sarsa(lambda)
 
 Sarsa(lambda) Algorithm
 
 Sarsa(lambda) Gridworld Example

 Off-Policy Learning
 
 Importance Sampling
 
 Q-learning
 
 Off-Policy Control with Q-learning
 
 Q-learning Control Algorithm (SARSAMAX)

 Relationship Between DP and TD

## Value Function Approximation

## Policy Gradient Methods

![image](https://github.com/user-attachments/assets/579dcaad-e150-4017-a00b-61305f3bdb1b)

`Policy Objective Functions:`

$`J_1(\theta) = V_{\pi_\theta}(s_1) = \mathbb{E}_{\pi_\theta} [v_1]`$ 

$`J_{\text{avV}}(\theta) = \sum_{s} d_{\pi_\theta}(s) V_{\pi_\theta}(s)`$

$`J_{\text{avR}}(\theta) = \sum_{s} d_{\pi_\theta}(s) \sum_{a} \pi_\theta(s, a) R^a_s`$


`Score Function`:

Likelihood ratios exploit the following identity

$`\nabla_{\theta} \pi_{\theta}(s, a) = \pi_{\theta}(s, a) 
\frac{\nabla_{\theta} \pi_{\theta}(s, a)}{\pi_{\theta}(s, a)} 
= \pi_{\theta}(s, a) \nabla_{\theta} \log \pi_{\theta}(s, a)`$

The score function is $`\nabla_{\theta} \log \pi_{\theta}(s, a)`$

### Compatible Function Approximation

Value function approximator is compatible to the policy

$`\nabla_w Q_w(s, a) = \nabla_{\theta} \log \pi_{\theta}(s, a)`$

Value function parameters w minimise the mean-squared error

$`\varepsilon = \mathbb{E}_{\pi_{\theta}} \left[ \left(Q_{\pi_{\theta}}(s, a) - Q_w(s, a) \right)^2 \right]`$

Then the policy gradient is exact,

$`\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}} 
\left[ \nabla_{\theta} \log \pi_{\theta}(s, a) Q_w(s, a) \right]`$

## Integrating Learning and Planning

## Exploration and Exploitation
