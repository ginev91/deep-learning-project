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

$$
16 \times 16,
$$

while MNIST images are

$$
28 \times 28.
$$

USPS images are therefore resized using **bilinear interpolation** to

$$
28 \times 28.
$$

### Pixel Normalization

Pixel values are normalized from

$$
[0,1]
$$

to

$$
[-1,1]
$$

using

$$
x_{\text{norm}}
=
\frac{x-0.5}{0.5}.
$$

## Dataset Properties

| Property | Source Domain (MNIST) | Target Domain (USPS) |
|-----------|----------------------|----------------------|
| Data Type | Synthetic handwritten digits | Scanned postal digits |
| Native Resolution | $28 \times 28$ | $16 \times 16$ |
| Network Input | $1 \times 28 \times 28$ | $1 \times 28 \times 28$ |
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
