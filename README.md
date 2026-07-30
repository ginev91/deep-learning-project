# Domain Adaptation in Computer Vision: Deep Feature Alignment

**Deep Feature Covariance Alignment for Handwritten Digit Classification (MNIST → USPS)**

**Author:** Aleksandar Ginev  
**Project Type:** Deep Learning / Computer Vision / Domain Adaptation  

---

## What this project is about

Standard deep learning models assume that training and deployment data come from identical distributions (**i.i.d. assumption**). In real-world computer vision systems, this assumption frequently collapses due to **domain shift** (variations in lighting, background noise, resolution, or sensor characteristics):

$$P_S(X) \neq P_T(X) \quad \text{while assuming} \quad P_S(Y\vert{}X) \approx P_T(Y\vert{}X)$$

When a Convolutional Neural Network (CNN) trained on clean, centered images is evaluated on noisy or lower-resolution images, its high-level latent features fail to generalize.

This project implements **Unsupervised Domain Adaptation (UDA)** between two classic digit recognition benchmarks:
* **Source Domain ($D_S$):** **MNIST** (70,000 clean, centered $28 \times 28$ grayscale images) with full class labels.
* **Target Domain ($D_T$):** **USPS** (9,298 noisy, $16 \times 16$ grayscale scans from real mail envelopes) with **no labels** available during model adaptation.

The primary objective is to adapt a deep neural network's bottleneck feature representations using **Deep CORAL (Deep Correlation Alignment)** (*Sun & Saenko, 2016*), training the network to extract domain-invariant features across both datasets simultaneously.

---

## Data & Deep Learning Pipeline

### Datasets
- **MNIST (Source):** $28 \times 28$ resolution, high contrast, uniform stroke thickness.
- **USPS (Target):** Natively $16 \times 16$, lower resolution, varying stroke weights, and spatial noise.

```text
[MNIST Source Images (28x28)] ───► [Shared Convolutional Backbone] ───► Source Latent Features (f_S) ──┐
                                                                                                        ├── Deep CORAL Loss (C_S vs C_T)
[USPS Target Images (28x28)] ───► [Shared Convolutional Backbone] ───► Target Latent Features (f_T) ──┘
