# Domain Adaptation in Computer Vision: Deep Feature Alignment

**Deep Feature Covariance Alignment for Handwritten Digit Classification (MNIST → USPS)**

**Author:** Aleksandar Ginev  
**Project Type:** Deep Learning / Computer Vision / Domain Adaptation  

---

## What this project is about

Standard deep learning models assume that training and deployment data come from identical distributions (**i.i.d. assumption**). In real-world computer vision systems, this assumption frequently collapses due to **domain shift** (variations in lighting, background noise, resolution, or sensor characteristics):

Pₛ(X) ≠ Pₜ(X), while Pₛ(Y|X) ≈ Pₜ(Y|X)

When a Convolutional Neural Network (CNN) trained on clean, centered images is evaluated on noisy or lower-resolution images, its high-level latent features fail to generalize.

This project implements **Unsupervised Domain Adaptation (UDA)** between two classic digit recognition benchmarks:
* **Source Domain (D_S):** **MNIST** (70,000 clean, centered 28 × 28 grayscale images) with full class labels.
* **Target Domain (D_T):** **USPS** (9,298 noisy, 16 × 16 grayscale scans from real mail envelopes) with **no labels** available during model adaptation.

The primary objective is to adapt a deep neural network's bottleneck feature representations using **Deep CORAL (Deep Correlation Alignment)** (*Sun & Saenko, 2016*), training the network to extract domain-invariant features across both datasets simultaneously.

---

## Data & Deep Learning Pipeline

### Datasets
- **MNIST (Source):** 28 × 28 resolution, high contrast, uniform stroke thickness.
- **USPS (Target):** Natively 16 × 16, lower resolution, varying stroke weights, and spatial noise.

```text
[MNIST Source Images (28x28)] ───► [Shared Convolutional Backbone] ───► Source Latent Features (f_S) ──┐
                                                                                                        ├── Deep CORAL Loss (C_S vs C_T)
[USPS Target Images (28x28)] ───► [Shared Convolutional Backbone] ───► Target Latent Features (f_T) ──┘
```
---

---

# 1. Project Setup & Environment Requirements

To ensure full reproducibility across different operating systems (macOS, Linux, and Windows), the project is implemented using **Python** and **PyTorch**. The environment is configured to support both CPU and GPU execution while maintaining deterministic behavior across multiple runs.

## Prerequisites & Dependencies

Install the required libraries using:

```bash
pip install torch torchvision numpy matplotlib pandas scikit-learn
```

## Global Configuration

The following configuration is applied before any experiments are executed:

- **Device Allocation:** Automatically selects CUDA (GPU) when available; otherwise falls back to CPU.
- **Reproducibility:** Random seeds are fixed (`seed = 42`) across PyTorch, NumPy, and Scikit-Learn.
- **SSL Bypass Setup:** A custom HTTPS context is configured to avoid SSL certificate verification issues when downloading benchmark datasets automatically.

---

# 2. Dataset Ingestion & Preprocessing

## Spatial & Intensity Alignment

To enable both datasets to share the same convolutional backbone, all images are transformed into a common input representation.

### Image Resizing

USPS images have a native resolution of

16 × 16,

while MNIST images are

28 × 28.

USPS images are therefore resized using **bilinear interpolation** to

28 × 28.

### Pixel Normalization

Pixel values are normalized from

[0,1]

to

[-1, 1]

using

x_norm
=
x-0.50.5.

## Dataset Properties

| Property | Source Domain (MNIST) | Target Domain (USPS) |
|-----------|----------------------|----------------------|
| Data Type | Synthetic handwritten digits | Scanned postal digits |
| Native Resolution | 28 × 28 | 16 × 16 |
| Network Input | 1 × 28 × 28 | 1 × 28 × 28 |
| Labels | Fully labeled | Unlabeled during training |
| Batch Size | 128 | 128 |

---

# 3. Visualizing the Domain Shift (Domain Gap)

Before training any model, representative samples from both datasets are inspected to understand the visual differences between the domains.

### MNIST

- High image contrast
- Clean background
- Uniform stroke thickness
- Centered handwritten digits

### USPS

- Lower contrast
- Variable stroke thickness
- Slight background artifacts
- Greater writing variability

Although both datasets contain the same semantic classes (digits 0–9), these low-level image differences create a noticeable **domain shift**, which ultimately propagates into the learned feature representations of a deep neural network.

---

# 4. Untrained Feature Space Analysis (Initial Distribution Gap)

To visualize the initial domain discrepancy, batches from both datasets are passed through a randomly initialized CNN feature extractor.

The bottleneck feature representation is

z ∈ ℝ¹²⁸.

These high-dimensional vectors are projected into two dimensions using PCA and t-SNE.

---

## 4.1 Principal Component Analysis (PCA)

Principal Component Analysis performs a linear projection by preserving the directions of maximum variance.

The projection is computed as

X_PCA
=
XW_k.

---

## 4.2 t-Distributed Stochastic Neighbor Embedding (t-SNE)

Unlike PCA, t-SNE is a nonlinear dimensionality reduction technique that preserves local neighborhood structure.

It minimizes the Kullback–Leibler divergence between probability distributions in the original and projected spaces:

KL(P ∥ Q)
=
Σ_i
Σ_j
p_ij
log
p_ijq_ij.

### Key Observation

Even before training begins, the feature representations extracted from MNIST and USPS occupy two largely separated regions of the latent space, demonstrating a substantial distribution mismatch caused solely by differences in pixel statistics.

---

# 5. Source-Only Baseline Model Training

A baseline CNN is trained exclusively on the labeled MNIST dataset without applying any domain adaptation.

---

## 5.1 Deep CNN Architecture

```text
Input (1×28×28)
   │
   ├── Conv2D(1 → 32, 3×3, padding=1)
   │      ↓
   │   BatchNorm
   │      ↓
   │     ReLU
   │      ↓
   │  MaxPool(2×2)
   │
   ├── Conv2D(32 → 64, 3×3, padding=1)
   │      ↓
   │   BatchNorm
   │      ↓
   │     ReLU
   │      ↓
   │  MaxPool(2×2)
   │
   ├── Flatten
   │
   ├── Linear(3136 → 128)
   │      ↓
   │     ReLU
   │
   └── Bottleneck Feature Vector
         f ∈ ℝ¹²⁸
               │
               ▼
        Linear(128 → 10)
               │
               ▼
      Cross-Entropy Loss
```

---

## 5.2 Baseline Loss Function

The baseline model minimizes the standard cross-entropy loss over the labeled source dataset:

L_cls
=
-1B
Σ_i=1^B
Σ_c=0^9
y_i,c
log(\hat y_i,c).

---

## 5.3 Quantitative Baseline Results

| Dataset | Accuracy |
|----------|----------|
| MNIST (Source) | **98.64%** |
| USPS (Target) | **81.76%** |

**Performance Drop**

98.64% − 81.76% = 16.88%.

This large decrease demonstrates that excellent source-domain performance does not necessarily translate into good target-domain generalization.

---

# 6. Deep CORAL (Correlation Alignment)

To reduce the domain discrepancy without using target labels, the project incorporates **Deep CORAL** (Sun & Saenko, 2016).

---

## 6.1 Mathematical Formulation

For a batch of deep features

D ∈ ℝᴮˣᵈ

the covariance matrix is

 C
=
1B-1
\left(
 D^\top D
-
1B
(1^\top D)^\top
(1^\top D)
\right).

The CORAL loss minimizes the squared Frobenius distance between the source and target covariance matrices:

 L_CORAL
=
14d^2
\left\|
 C_S- C_T
\right\|_F^2.

Equivalently,

 L_CORAL
=
14d^2
Σ_i=1^d
Σ_j=1^d
\left(
C_S,ij
-
C_T,ij
\right)^2.

For this project,

d = 128.

---

## 6.2 Joint Optimization Strategy

The total training objective combines classification and covariance alignment:

 L_total
=
 L_CrossEntropy
+
λ
 L_CORAL.

The weighting coefficient is

λ = 10.0

which balances source-domain classification accuracy with domain-invariant feature learning.

---

# 7. Results, Confusion Matrices & Feature Alignment Analysis

---

## 7.1 Quantitative Performance Comparison

| Model | MNIST Accuracy | USPS Accuracy | Improvement |
|--------|---------------|---------------|-------------|
| Source-Only CNN | **98.64%** | **81.76%** | Baseline |
| Deep CORAL | **98.64%** | **82.51%** | **+0.75%** |

Although the numerical improvement appears modest, it is achieved **without using any USPS labels during training**, making it a meaningful gain under the unsupervised domain adaptation setting.

---

## 7.2 Confusion Matrix Analysis

The confusion matrices reveal class-specific improvements after applying Deep CORAL.

### Baseline CNN

- Confuses visually similar digits more frequently.
- Misclassifies many handwritten **3s** as **5s**.
- Confuses **7s** with **1s** due to reduced resolution.

### Deep CORAL

- Produces more stable feature representations.
- Reduces inter-class confusion.
- Improves robustness to varying stroke thickness and writing styles.

---

## 7.3 Post-Adaptation Feature Alignment (t-SNE)

The bottleneck features

f ∈ ℝ¹²⁸

are projected once again using t-SNE after training.

Compared with the initial visualization:

- Source and target feature distributions overlap substantially.
- USPS samples move closer to the corresponding MNIST class clusters.
- Domain-specific clustering is significantly reduced while preserving semantic class separation.

These observations indicate that Deep CORAL successfully learns a more domain-invariant feature representation while maintaining discriminative information necessary for accurate digit classification.
## 8. Conclusion & Critical Analysis

### Key Takeaways
- **Unsupervised Alignment:** Deep CORAL successfully aligns second-order statistics (covariances) between domains without using target labels, improving target accuracy (81.76% → 82.51%).
- **Efficiency:** Unlike adversarial methods (DANN), CORAL requires no extra discriminator network, making it computationally light and training-stable.

### Limitations & Next Steps
- **Second-Order Bound:** CORAL ignores higher-order feature statistics, which limits alignment on extreme domain gaps.
- **Future Directions:** Combining covariance alignment with Maximum Mean Discrepancy (MMD) or evaluating on complex benchmarks (e.g., SVHN → MNIST).

---

## 9. References & Reproducibility

### Academic References
1. **Sun, B., & Saenko, K. (2016).** *Deep CORAL: Correlation Alignment for Deep Domain Adaptation.* ECCV.
2. **LeCun, Y., et al. (1998).** *Gradient-based learning applied to document recognition.* Proceedings of the IEEE.
3. **Hull, J. J. (1994).** *A database for handwritten text recognition research (USPS).* IEEE TPAMI.

### Hyperparameters
| Parameter | Value |
| :--- | :--- |
| **Random Seed** | `42` |
| **Optimizer** | Adam (lr = 10⁻³) |
| **CORAL Weight (λ)** | `10.0` |
| **Feature Dimension (d)** | `128` |
| **Batch Size** | `128` |
