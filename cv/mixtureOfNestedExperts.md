# Efficient Model Scaling via Nested Experts (Conditional Computation) Google

## Motivation

- Modern **Vision Transformers (ViTs)** scale well in accuracy but poorly in **inference efficiency** and **memory footprint**.
- Every token (patch) is processed equally, even if many represent _uninformative background regions_.
- This results in **wasted computation** and **high GPU memory use**.

Goal:

- ⚡Improve model efficiency during both training & inference.
- Increase effective model capacity without increasing inference cost.
- Reduce large memory footprint by allocating computation _only where needed_.

---

## Key Idea: Nested Expert Layers with Token Routing

### Core Concept

Use a **router** to dynamically assign tokens to different experts based on importance.

### Architecture Overview

1. **Input tokens** → router network

2. Router predicts **importance probabilities** for each token

   - Example: 3 expert groups

     - **Expert 1:** Full (100% weights, most capable)
     - **Expert 2:** Half-size (50% weights)
     - **Expert 3:** Quarter-size (25% weights)

3. Each token’s router output gives a probability distribution across experts:

   $$
   p_i = \text{softmax}(W_r \cdot x_i)
   $$

   where (x*i) is the embedding of token \_i*, and (W_r) are router weights.

4. **Routing logic:**

   - Tokens with **high importance** → Expert 1 (full layer)
   - Tokens with **medium importance** → Expert 2
   - Tokens with **low importance** → Expert 3

5. Each expert processes _only its assigned tokens_ independently.

6. Outputs are **merged / aggregated** back into a unified sequence.

---

### Example Flow

```
Input Tokens → Router → Assign Probabilities
                ↓
      ┌─────────────────────────────┐
      │         Experts             │
      │  E1 (Full) | E2 (Half) | E3 (Quarter)  │
      └─────────────────────────────┘
                ↓
    Combine Outputs → Next ViT Layer
```

---

## Computation Saving Mechanism

1. **Selective Token Routing:**

   - Not all tokens are processed by all experts.
   - Tokens with background or low importance skip heavy computation.

2. **Nested Experts:**

   - Tokens not used by large experts are **downsampled / merged** and passed to smaller ones.
   - Ensures _every token_ contributes, but with proportional computation.

3. **Token Downsizing:**

   - Downsize or merge low-importance tokens before passing to smaller experts.
   - Achieves **quadratic cost reduction** in attention computation:
     $$
     \mathcal{O}(N^2) \to \mathcal{O}(N_r^2 + N_s^2 + N_t^2)
     $$
     where (N_r, N_s, N_t) are token counts for routed expert groups.

4. **Final Output Combination:**

   - Experts’ outputs concatenated or weighted averaged to form the unified embedding.

---

## Advantages

| Property                    | Explanation                                                                |
| --------------------------- | -------------------------------------------------------------------------- |
| **Conditional Computation** | Activates only a subset of model parameters per input.                     |
| **Increased Capacity**      | Larger effective model size (more experts) without higher per-sample cost. |
| **Dynamic Adaptation**      | Different images/patches activate different computation paths.             |
| **Memory Efficiency**       | Reduces redundant activations for uninformative tokens.                    |
| **Improved Throughput**     | Allows larger models to be used under same FLOPs budget.                   |

---

## Relation to Other Models

| Model / Approach                        | Mechanism                                            | Similarity                                             |
| --------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------ |
| **Mixture of Experts (MoE)**            | Route full inputs to expert networks based on gating | Same principle, but PLM-like nested ViT is token-level |
| **DynamicViT (ICCV 2021)**              | Drop unimportant tokens dynamically                  | Similar token-level pruning                            |
| **Token Merging (ToMe)**                | Merge similar patches to reduce token count          | Complementary to routing approach                      |
| **Nested ViT**                          | Multi-scale experts with token downsampling          | Closely related — similar hierarchical computation     |
| **Sparse Mixture-of-Attention (S-MoA)** | Apply attention only to top-K tokens per expert      | Mechanically similar to router-based nested experts    |

---

## Summary

| Aspect          | Description                                                             |
| --------------- | ----------------------------------------------------------------------- |
| **Idea**        | Nested expert layers that adaptively route tokens based on importance   |
| **Goal**        | Scale model capacity without scaling compute                            |
| **Routing**     | Router assigns tokens to different experts with varying compute budgets |
| **Computation** | Heavy experts for key regions; lightweight experts for background       |
| **Result**      | Efficient large models with preserved accuracy and lower inference cost |

---

**Conceptual Analogy:**
Think of this as a _“Transformer with an attention budget”_ — where only the most important patches are allowed to “spend” computation in large experts, while the rest get lightweight processing.

---
