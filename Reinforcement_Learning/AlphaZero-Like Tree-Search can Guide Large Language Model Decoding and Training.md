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
