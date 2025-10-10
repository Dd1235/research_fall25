# FACET — Fairness in Computer Vision Evaluation Benchmark (Meta AI)

## Overview

**FACET** (Fairness in Computer Vision Evaluation Benchmark) is a **dataset and benchmark** introduced by **Meta AI** to evaluate **fairness and bias** in computer vision models — particularly those trained on large-scale, web-scraped datasets like ImageNet-21K or LAION.

> Goal: Build a **systematic, transparent benchmark** to test how well vision models treat different demographic groups **consistently and equitably**.

---

## Motivation

- Modern vision models (ViT, CLIP, DINOv2, etc.) are trained on **massive, uncurated datasets** scraped from the internet.
- These datasets **inadvertently encode societal biases** (gender, race, age, attire, context).
- There was **no standardized benchmark** to measure **representation bias** or **performance disparities** across demographic subgroups.
- **FACET** fills this gap — providing an **evaluation-only dataset** for **fairness testing**, not for training.

---

## Dataset Construction

### 1️⃣ Root Category Selection

- Started with the **"person"** synset (semantic category) in **WordNet**.
- Collected **child nodes** under “person” (e.g., _man_, _woman_, _worker_, _athlete_, _bride_, _student_, etc.).

### 2️⃣ Filtering by Overlap with ImageNet

- Cross-referenced these WordNet “person” child categories with **ImageNet classes**.
- Kept only those overlapping with ImageNet-1K/21K classes to ensure availability of images and model compatibility.

### 3️⃣ Image Gathering & Verification

- Gathered a curated set of **images depicting people** under these categories.
- Manually verified samples for demographic diversity (age, gender presentation, skin tone, and geographic diversity).

### 4️⃣ Annotation for Fairness Attributes

Each image is annotated with:

- **Demographic attributes:** perceived gender, age range, skin tone, etc.
- **Contextual features:** background, pose, clothing, lighting, occlusion.
- **Image labels:** class from the “person-related” subset.

---

## Benchmark Design

FACET is designed not for training but for **evaluation** — to test if a model:

1. Maintains **consistent accuracy** across demographics.
2. Avoids **systematic misclassification** or **underrepresentation**.
3. Exhibits **equal calibration** and **embedding similarity** across groups.

The benchmark measures metrics like:

- **Per-group accuracy**
- **False positive / false negative disparity**
- **Representation bias in embeddings**
- **Fairness calibration gap**

---

## Usage in Research

- FACET acts as a **drop-in evaluation suite** for models like CLIP, DINOv2, and LLaVA.
- It helps detect issues like:

  - Underperformance on darker skin tones or underrepresented age groups.
  - Gender-coded associations in embeddings (e.g., associating “doctor” with men).
  - Overfitting to Western-centric cultural imagery.

---

## Example Metric: Group Disparity

If ( A*g ) is accuracy for group ( g ), and ( A*{\text{overall}} ) is overall accuracy, then:

$$
\text{Fairness Gap} = \max_g | A_g - A_{\text{overall}} |
$$

A smaller fairness gap indicates **more equitable model performance**.

---

## Key Insights from FACET Paper (Meta AI, 2024)

- Even **state-of-the-art ViTs and CLIP-style models** exhibit notable fairness gaps.
- **Self-supervised models** (e.g., DINOv2) sometimes generalize better but are not bias-free.
- **Bias correlates** with training dataset composition — e.g., over-representation of specific demographics or professions.
- FACET provides **controlled, measurable evidence** of these disparities, guiding dataset and model auditing.

---

## Summary Table

| Aspect             | Description                                              |
| ------------------ | -------------------------------------------------------- |
| **Name**           | FACET (Fairness in Computer Vision Evaluation Benchmark) |
| **Developed by**   | Meta AI                                                  |
| **Purpose**        | Evaluate fairness and bias in vision models              |
| **Root Source**    | WordNet “person” subtree                                 |
| **Filtered Using** | Overlap with ImageNet classes                            |
| **Annotations**    | Gender, age, skin tone, context, class                   |
| **Use Case**       | Fairness testing, bias auditing, demographic evaluation  |
| **Key Finding**    | Large models still show systematic fairness gaps         |

---

## Takeaway

FACET provides a **rigorous, reproducible** benchmark to assess **demographic fairness** in vision models — encouraging the community to not just chase accuracy, but also **equity and accountability** in AI perception systems.
