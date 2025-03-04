# AlphaZero-Like Tree-Search can Guide Large Language Model Decoding and Training

https://openreview.net/pdf?id=C4OpREezgj

![Screenshot (171)](https://github.com/user-attachments/assets/43998cda-4b4e-4af9-9db6-5712e5cb15d2)

we use language model πθ as the policy to sample generations using the task training dataset. With true
label or a given reward function in training data, a set of
sampled tuple $`D_{\text{train}} = \{(x_j, y_j, r_j)\}_j`$ can be
obtained, where xj is the input tex   $`y_j = s_j^{0:T_j-1}`$  1
is the output text of Tjsteps and $`r_j = R(y_j | x_j)`$ is the ground-truth
reward. Similar to the critic training in most RL algorithms,
we construct the value target z
jt by TD-λ (Sutton, 1988) or MC estimate (Sutton & Barto, 2018) on each single step t .The value network is optimized by mean squared error:
$`L(\varphi) = \mathbb{E}_{D} \left[ \sum_{j=1}^{T-1} \sum_{t=0}^{j-1} \left\| v_\varphi(s_{j, 0:t} | x_j) - z_{j, t} \right\|^2_2 \right]`$


The ORM $`\hat{r}_\varphi(y_{0:T-1} | x_{0:L-1})`$ is learned with the same objective. Training an accurate value function and ORM is quite crucial for the tree-search process as they provide the main guidance. We will further illustrate how to learn a reliable value function and ORM in our experiment section.

$`a \sim \frac{N(s_t, a)^{1/\tau}}{\sum_b N(s_t, b)^{1/\tau}}`$

ORM-Max. Given an outcome reward model, the aggregation can choose the answer f with maximum final reward $`f^* = \arg\max_f \sum_{y_j} \mathbb{1}_{\text{final ans}(y_j) = f}`$ ORM-Vote. Given an outcome reward model, the aggregation can choose the answer f with the sum of rewards,
namely $`f^* = \text{final ans}\left(\arg\max_{y_j} \hat{r}_\varphi(y_j | x_j)\right)`$


##  Enhancing LLM Training with Tree Search
In section 3.2 we discuss how tree-search can guide LLM’s
decoding process during inference time. Such guidance
leads to a better decoding strategy and improves the performance of given tasks. In other words, tree-search guidance
can serve as a policy improvement operator. Based on this,
we propose a new training and finetuning paradigm.
Assume we have an initial LLM policy πθold (trained by
conducting supervised finetuning over the original training
set) and initial LLM value and ORM: vϕold , rˆϕold (trained by
Equ. 1 from sampling the original training questions), we
can have the following iterative process:
Policy Improvement: We conduct tree-search over training
set based on πθold , vϕold , and rˆϕold to obtain improved generations, resulting in the augmented dataset D and also the
filtered positive examples D+.
Policy Distillation: With the tree-search-improved dataset
D+, by imitating the tree-search positive trajectories, LLM
policy can be further improved to πθnew with supervised loss
$`L(\theta) = \mathbb{E}_{(x_j, y_j) \sim D^+} \left[ -\log \pi_\theta(y_j | x_j) \right]`$
