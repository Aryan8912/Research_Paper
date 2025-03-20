# PsiPhi-Learning: Reinforcement Learning with Demonstrations using Successor Features and Inverse Temporal Difference Learning

![Screenshot (219)](https://github.com/user-attachments/assets/c9d50c03-1a09-4ecf-b4bc-89a332f1cef7)

Definition 1 (Cumulants and Preferences). The (one-step) rewards are decomposed into task-agnostic cumulants Φ(s, a) ∈ R d, and task-specific preferences w ∈ R d :
$`R_w(s, a) = \Phi(s, a) \cdot w`$
The preferences w are a representation of a possible goal in the world C, in the sense that each w gives rise to a task Mw = hC, Rwi. We use ‘task’, ‘goal’, and ‘preferences’ interchangeably when context makes it clear whether we are referring to w itself, or the corresponding Mw or Rw. The action-value function for a policy π in Mw is then a function of the preferences w and the π’s successor features.

Definition 2 (Successor Features). For a given discount factor γ ∈ [0, 1), policy π and cumulants Φ(s, a) ∈ R d , the successor features (SFs) for a state s and action a are:
$`\Psi_{\pi}(s, a), \quad \mathbb{E}_{C, \pi} \left[ \sum_{t=0}^{\infty} \gamma^t \Phi(s_t, a_t) \,\middle|\, s_0 = s, a_0 = a \right]`$
The i-th component of Ψπ
(s, a) gives the expected discounted sum of Φ(s, a)’s i-th component, when starting from state s, taking action a and then following policy π. Intuitively, cumulants Φ can be seen as a vector-valued reward function and SFs Ψπ the corresponding vector-valued state-action value function for policy π. An action-value function is then given by the dot product of the preferences w and π’s SFs:

$`Q_{\pi, w}(s, a) = \Psi_{\pi}(s, a) \cdot w`$

Note that if we have Ψπ, the value of π for a new preference w0 can be easily computed. This property allows the successor features of a set of policies to be repurposed for accelerating policy updates, as follows.


Definition 3 (Generalised Policy Improvement). Given a set of policies Π = {π1, . . . , πK} and a task with reward function R, generalised policy improvement (GPI) is the definition of a policy π0 such that

$`Q_{\pi_0, R}(s, a) \geq \sup_{\pi \in \Pi} Q_{\pi, R}(s, a), \quad \forall s \in S, a \in A`$

Provided the SFs of a set of policies, i.e., {Ψπk}K k=1, we can apply GPI to derive a new policy π0 whose performance on a task w is no worse that the performance of any of π ∈ Π on the same task, given by

$`\pi_0(s) = \arg\max_{a} \max_{\pi \in \Pi} \Psi_{\pi}(s, a) \cdot w`$

## Inverse Temporal Difference Learning

Given demonstrations without rewards, D, we model the agents that generated the data (i.e., blue nodes in Figure 1) as soft-optimal for an unknown task. In particular, the k-th agent’s policy is soft-optimal under task wk and is given by

$`\pi_k(a \mid s) = \frac{\exp(\Psi_{\pi_k}(s, a) \cdot w_k)}
{\sum\limits_{a} \exp(\Psi_{\pi_k}(s, a) \cdot w_k)}, \quad \forall s \in S, a \in A`$

We choose to represent the action-value functions of the other agents with their SFs and preferences to enable GPI, and to expose task- and policy-agnostic structure in the form of shared cumulants Φ. The k-th agent’s successor features are temporally consistent with these cumulants Φ

$`\Psi_{\pi_k}(s, a) = \Phi(s, a) + \gamma \mathbb{E}_{C, \pi_k} \left[ \Psi_{\pi_k}(s', \pi_k(s')) \right]`$

## Behavioural cloning loss

Given demonstrations generated only by the k-th agent, i.e., Dk ⊂ D, we train its successor features θΨk and the preferences wk by minimising the negative log-likelihood of the demonstrations

$`L_{\text{BC-Q}}(\theta_{\Psi_k}, w_k) = -\mathbb{E}_{\tau \sim D_k} 
\log \frac{\exp(\Psi(s_t, a_t; \theta_{\Psi_k}) \cdot w_k)}
{\sum\limits_{a} \exp(\Psi(s_t, a; \theta_{\Psi_k}) \cdot w_k)}`$

## Inverse temporal difference loss

$`L_{\text{ITD}}(\theta_{\Phi}) = \mathbb{E}_{(s_t, a_t, s_{t+1}, a_{t+1}, k) \sim D_k} 
\left[ \Phi(s_t, a_t; \theta_{\Phi}) \leftarrow 
\gamma \Psi(s_{t+1}, a_{t+1}; \tilde{\theta}_{\Psi_k}) - \Psi(s_t, a_t; \theta_{\Psi_k}) \right]_{\text{stop gradient}}`$

## Reward loss

$`L_R(\theta_{\Phi}, w_{\text{ego}}) = \mathbb{E}_{(s, a, r_{\text{ego}}) \sim B_k} 
\left[ \Phi(s, a; \theta_{\Phi}) \cdot w_{\text{ego}} - r_{\text{ego}, k} \right]`$

## Temporal difference learning

$`L_Q(\theta_{\Psi_{\text{ego}}}) = \mathbb{E}_{(s, a, s', r_{\text{ego}}) \sim B_k} 
\left[ \Psi(s, a; \theta_{\Psi_{\text{ego}}}) \cdot w_{\text{ego}} - r_{\text{ego}} - \gamma \max_{a'} \Psi(s', a'; \tilde{\theta}_{\Psi_{\text{ego}}}) \cdot w_{\text{ego}} \right]_{\text{stop-gradient}}`$

$`L_{\text{TD}-\Psi}(\theta_{\Psi_{\text{ego}}}) = 
\mathbb{E}_{(s, a, s', a') \sim B_k} 
\left[ \Psi(s, a; \theta_{\Psi_{\text{ego}}}) - \Phi(s, a; \tilde{\theta}_{\Phi}) - \gamma \Psi(s', a'; \theta_{\Psi_{\text{ego}}}) \right]_{\text{stop-gradient}}`$

## GPI behavioural policy

$`\pi_{\text{ego}}(s) = \arg\max_{a} \max_{\theta_{\Psi}} 
\Psi(s, a; \theta_{\Psi}) \cdot w_{\text{ego}}`$

## Theorem 1. (Informal statement.)
Let π∗ be the optimal policy for the ego task w 0 and let π be the GPI policy obtained from {Q˜πi }, with δr, δΨ the reward and successor feature approximation errors. Then for all s, a

$`Q^*(s, a) - Q^{\pi}(s, a) \leq 
\frac{2}{1 - \gamma} 
\left( 
\phi_{\max} \min_j \| w_0 - w_j \|  + 2\delta_r + \| w_0 \| \delta_{\Psi} + \frac{\delta_r}{1 - \gamma} \right)`$

![Screenshot (220)](https://github.com/user-attachments/assets/89f2e035-cfb2-4b4f-b1c4-d77c598223f7)

![Screenshot (221)](https://github.com/user-attachments/assets/f274721e-1ab6-4f06-bb61-b1577c42e203)
