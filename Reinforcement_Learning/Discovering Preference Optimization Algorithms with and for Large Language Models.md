# Discovering Preference Optimization Algorithms with and for Large Language Models

https://arxiv.org/pdf/2406.08414

![Screenshot (168)](https://github.com/user-attachments/assets/b3988ef4-df0f-4260-a7f9-b4a2636396b4)

Pre-Trained Language model policy $`πθ`$ and dataset $`D = \{(x_i, y_i^w, y_i^l)\}_{i=1}^{N}`$ with prompts x and preference-ranked completions $`y_w`$ $`y_l`$ 
 $`\max_{\pi_\theta} \underbrace{\mathbb{E}_{y \sim \pi_\theta, x \sim P} [r_\varphi(y, x)]}_{\text{reward maximization}} - \beta \underbrace{\mathrm{KL}(\pi_\theta, \pi_{\text{ref}})}_{\text{regularization}}`$

$`\max_{\pi_\theta} \mathbb{E}_{y \sim \pi_\theta, x \sim P} \left[ 
\underbrace{r_\varphi(y, x)}_{\text{reward}} + \beta \underbrace{\log \pi_{\text{ref}}(y|x)}_{\pi_{\text{ref}} \text{ regularization}}\right] + \beta \underbrace{H(\pi_\theta)}_{\text{policy entropy}}`$


$`\min_{\pi_\theta} \mathbb{E}_{(y_w, y_l, x) \sim D} 
\left[ 
f\left( 
\beta \cdot 
\left( 
\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right)\underbrace{r_\varphi(y_w, x) - r_\varphi(y_l, x)}_{\text{reward difference}}\right]
`$

$`\rho = \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}`$

$`\pi^*(y|x) = Z(x)^{-1} \pi_{\text{ref}}(y|x) \exp\left(\beta^{-1} r_\varphi(y, x)\right)`$

![Screenshot (170)](https://github.com/user-attachments/assets/e3ce7b98-6bf4-44c9-a662-99f15728f7e7)

$`\{\log \pi_\theta(y_w|x), \log \pi_{\text{ref}}(y_w|x), \log \pi_\theta(y_l|x), \log \pi_{\text{ref}}(y_l|x)\}`$

$`f_{\text{lrml}}(\beta \rho) = \left( \sigma\left(\frac{\beta \rho}{\tau}\right) - 1 \right) \cdot f_{\text{dpo}}(\beta \rho) + \sigma\left(\frac{\beta \rho}{\tau}\right) \cdot f_{\text{exp}}(\beta \rho)`$

$`= (1 - \sigma\left(\frac{\beta \rho}{\tau}\right)) \cdot \log(1 + \exp(-\beta \rho)) + \sigma\left(\frac{\beta \rho}{\tau}\right) \cdot \exp(-\beta \rho)`$
