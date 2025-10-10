# Group Relative Policy Optimization (GRPO)

## Background

Large Language Models (LLMs) are often fine-tuned with **Reinforcement Learning from Human Feedback (RLHF)** to align behavior with desired outcomes.

A key algorithm in RLHF pipelines is **PPO — Proximal Policy Optimization (Schulman et al., 2017)**.

PPO stabilizes policy updates using a clipped objective that prevents large gradient steps from causing catastrophic divergence.

---

## PPO: Core Mechanism

### Training Setup

```
Prompt → Policy (LLM) → Response
              ↓
       Reward Model → Reward r
              ↓
        Value Model → Value v (expected reward)
```

### PPO Objective

The PPO loss encourages the model to increase the probability of better responses (high advantage) and decrease it for worse ones.

Let:

- ( \pi\_\theta ): current (new) policy
- ( \pi\_{\text{old}} ): reference / previous policy
- ( r_t = R_t ): reward from reward model
- ( V_t ): baseline value predicted by value model
- ( A_t = R_t - V_t ): advantage (how much better/worse than expected)
- ( \hat{r}_t = \frac{\pi_\theta(o*t)}{\pi*{\text{old}}(o_t)} ): probability ratio

Then the PPO objective is:

$$
\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_t \left[
\min\left(
\hat{r}_t A_t,
\text{clip}(\hat{r}_t, 1 - \epsilon, 1 + \epsilon) A_t
\right)
\right]

* \beta , D_{\text{KL}}\big(\pi_\theta ,|, \pi_{\text{ref}}\big)
$$

- **Clipping** keeps updates within ±ε to stabilize training.

- **KL penalty** ensures the model doesn’t drift too far from the reference model (helps retain language quality).

- **Value model** provides a running estimate of expected rewards to reduce variance.

---

## Limitations of PPO in LLM Training

| Limitation                     | Description                                                                   |
| ------------------------------ | ----------------------------------------------------------------------------- |
| **Memory-heavy**               | Requires an extra **value model** (often same size as policy model).          |
| **Slow on inference hardware** | Not optimized for LLM fine-tuning on **inference-class GPUs (A100s, H100s)**. |
| **High variance**              | Reward normalization needed; still unstable for multi-sample batch training.  |

---

## 🚀 Group Relative Policy Optimization (GRPO)

**GRPO** is a modern, more efficient variant of PPO introduced by Meta and others for large-scale RLHF training.

It removes the **value model** entirely, instead estimating the baseline (expected reward) directly from **a group of sampled responses**.

### Motivation

- Reduce **memory footprint** and **compute cost**.
- Make RLHF feasible on **inference-type hardware** (optimized for forward passes).
- Retain **stability** and **alignment quality** comparable to PPO.

---

## How GRPO Works

### Step 1. Sampling Multiple Responses

From a single prompt, sample **G** candidate responses:

$$
{ y_1, y_2, \dots, y_G } \sim \pi_\theta(y|x)
$$

Each response ( y_i ) is scored by the **reward model**, producing rewards ( r_i ).

---

### Step 2. Compute Group-Based Baseline

Instead of predicting value (V_t) from a separate network, GRPO computes the **average reward** of the group as a baseline:

$$
\bar{r} = \frac{1}{G} \sum_{i=1}^{G} r_i
$$

This serves the same role as the value estimate.

---

### Step 3. Compute Advantage

For each sample:

$$
A_i = r_i - \bar{r}
$$

The advantages are **normalized** to zero mean and unit variance across the batch to stabilize updates:

$$
A_i \leftarrow \frac{A_i - \mu_A}{\sigma_A}
$$

---

### Step 4. Compute Policy Ratio

For each token sequence ( y_i ):

$$
\rho_i = \frac{\pi_\theta(y_i | x)}{\pi_{\text{old}}(y_i | x)}
$$

This ratio reflects the _change in model confidence_ for that response.

- If ( \rho_i > 1 ): new model more confident (probability increased)
- If ( \rho_i < 1 ): new model less confident (probability decreased)

---

### Step 5. GRPO Objective

The **optimization objective** is similar to PPO but without the value term inside the advantage:

$$
\mathcal{L}_{\text{GRPO}}(\theta)
= \mathbb{E}_i \Big[
\min\big(
\rho_i A_i,
\text{clip}(\rho_i, 1 - \epsilon, 1 + \epsilon) A_i
\big)
\Big]

* \beta , D_{\text{KL}}\big(\pi_\theta ,|, \pi_{\text{ref}}\big)
$$

- The **KL penalty** still discourages large deviation from the reference model but is **not included inside** the advantage.

- This allows **clearer gradient flow** and reduces variance.

---

## ⚖️ PPO vs GRPO

| Feature                | PPO                         | GRPO                       |
| ---------------------- | --------------------------- | -------------------------- |
| **Baseline**           | Value model                 | Group average reward       |
| **Extra model needed** | Yes (value net)             | No                         |
| **Memory usage**       | High                        | Lower (~½)                 |
| **Hardware**           | Training-class (A100, TPUs) | Inference-class (L4, T4)   |
| **Variance**           | Moderate                    | Lower (with normalization) |
| **KL placement**       | Inside advantage term       | Outside advantage term     |
| **Stability**          | Good, needs value model     | Comparable or better       |

---

## Conceptual Summary

**PPO intuition:**

> “Increase probability of better-than-expected responses, decrease probability of worse ones — but only a little at a time.”

**GRPO intuition:**

> “Do the same, but estimate ‘expected’ reward directly from the group of generated responses instead of using a separate model.”

This change:

- Greatly simplifies pipeline (no critic/value network),
- Makes the method **scalable to large LLMs**,
- Enables training using **inference-optimized compute** (high batch forward passes, no backprop in critic).

---

## Final Objective Breakdown

$$
\mathcal{L}*{\text{GRPO}} =
\underbrace{\mathbb{E} \big[ \min(\rho A, \text{clip}(\rho, 1-\epsilon, 1+\epsilon)A ) \big]}*{\text{Policy Clipping (like PPO)}}
;-;
\underbrace{\beta D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}})}_{\text{Stability Penalty}}
$$

Where:

- ( A = r - \bar{r} ) (group advantage)
- ( \beta ) tunes model drift
- ( \epsilon ) is clip threshold (usually 0.1–0.2)

---

## 🧩 Outcome vs Process Supervision

- **Outcome Supervision:** Train on final response quality only (classic RLHF).
- **Process Supervision:** Provide rewards at intermediate reasoning steps, improving chain-of-thought alignment and interpretability.
- GRPO can integrate both since it reuses group-based relative rewards for any time step.

---

## Practical Impact

- 🧮 **No critic model** → half memory, faster training.
- ⚙️ **Inference-class GPUs** can be used for fine-tuning.
- 📈 **Comparable or better alignment** quality than PPO in open benchmarks.
- 🔄 **KL stabilization** ensures language fluency isn’t degraded.

---

## 🧭 Summary Table

| Concept                  | PPO                     | GRPO                         |
| ------------------------ | ----------------------- | ---------------------------- |
| **Objective type**       | Actor–critic            | Actor-only (relative)        |
| **Advantage estimation** | Reward – Value          | Reward – Mean Reward (group) |
| **KL term**              | Inside advantage        | Added outside                |
| **Value model**          | Required                | Removed                      |
| **Hardware Efficiency**  | High compute            | Inference-optimized          |
| **Supervision Type**     | Outcome or process      | Both supported               |
| **Stability**            | Strong but memory heavy | Stable and lightweight       |

---

## Takeaway

> **GRPO** simplifies RLHF by **removing the value model** and using **relative group rewards** as a baseline — achieving PPO-level alignment quality with **far lower memory and hardware cost**.

It represents a practical step toward **scalable alignment training** for massive foundation models using **inference-type compute resources** (e.g., L4, T4 GPUs).

---
