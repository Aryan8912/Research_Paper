# Discovering Preference Optimization Algorithms with and for Large Language Models

https://arxiv.org/pdf/2406.08414

![Screenshot (168)](https://github.com/user-attachments/assets/b3988ef4-df0f-4260-a7f9-b4a2636396b4)

Pre-Trained Language model policy $`πθ`$ and dataset $`D = \{(x_i, y_i^w, y_i^l)\}_{i=1}^{N}`$ with prompts x and preference-ranked completions $`y_w`$ $`y_l`$ 
 $`\max_{\pi_\theta} \underbrace{\mathbb{E}_{y \sim \pi_\theta, x \sim P} [r_\varphi(y, x)]}_{\text{reward maximization}} - \beta \underbrace{\mathrm{KL}(\pi_\theta, \pi_{\text{ref}})}_{\text{regularization}}`$
