
# Project Implementation Plan

**Goal:** Build an **unsupervised image-classification pipeline** for the **STL-10** data set by

1. pre-training a **ResNet-50** backbone with  **contrastive learning (SimCLR-style)** ,
2. extracting deep features,
3. running **PCA → K-means** ( *K = 10* known from STL-10), and
4. reporting clustering **and** “pseudo-classification” scores.

---

## 1. Environment & Resources

| Item                      | Spec / Choice                                                                            |
| ------------------------- | ---------------------------------------------------------------------------------------- |
| **Hardware**        | Laptop RTX 4060 (8 GB VRAM) + i9-13900HX + 16 GB DDR5                                    |
| **Frameworks**      | PyTorch 2.x, torchvision, PyTorch-Lightning (optional), cuML (GPU K-means), scikit-learn |
| **Mixed Precision** | `torch.cuda.amp`throughout to fit larger batch sizes                                   |
| **Reproducibility** | Fix `torch.manual_seed`, log with Weights & Biases                                     |

---

## 2. Data Preparation

| Split                     | Images            | Usage                                                                         |
| ------------------------- | ----------------- | ----------------------------------------------------------------------------- |
| **Unlabeled**       | 100 000           | Contrastive pre-training / clustering                                         |
| **Train (labeled)** | 5 000×10 classes | *never*shown to backbone in pre-training; later for evaluation/linear probe |
| **Test (labeled)**  | 8 000×10 classes | Final evaluation only                                                         |

```text
transforms_simclr =
  RandomResizedCrop(96) → RandomHorizontalFlip →
  ColorJitter(0.4,0.4,0.4,0.1) →
  RandomGrayscale(0.2) → GaussianBlur →
  ToTensor() → Normalize(mean,std)
```

---

## 3. Contrastive Self-Supervised Pre-training (SimCLR v2)

| Hyper-param     | Value                                                | Notes |
| --------------- | ---------------------------------------------------- | ----- |
| Backbone        | ResNet-50 (BN frozen)                                |       |
| Projection head | 2-layer MLP: 2048 → 512 → 128  (BN + ReLU between) |       |
| Batch size      | 256 (2×128 views) ← fits 8 GB VRAM with AMP        |       |
| Optimizer       | LARS (or AdamW)                                      |       |
| LR schedule     | Cosine decay; base LR = 4.8 / batch (≈ 1.2e-3)      |       |
| Temperature τ  | 0.2                                                  |       |
| Epochs          | 200 (≈ 1 h on RTX 4060)                             |       |
| Loss            | NT-Xent over 2 × batch                              |       |

 **Training loop** : per mini-batch → forward (views → h→ z) → NT-Xent loss → `loss.backward()` → `optimizer.step()`.

---

## 4. Feature Extraction

* **Freeze** ResNet-50 weights, **discard** projection head.
* Forward pass all images once → `features.npy` ∈ ℝN×2048.

  *Run time ≈ 3 min for 100 K.*

---

## 5. Dimensionality Reduction

```python
pca = PCA(n_components=100, svd_solver="randomized")
X_reduced = pca.fit_transform(features)
# retain ≳95 % cumulative variance
```

---

## 6. K-means Clustering

```python
km = cuML.KMeans(n_clusters=10, init="k-means++",
                 max_iter=300, n_init=10)
labels = km.fit_predict(X_reduced)   # 100 K pseudo-labels
centroids = km.cluster_centers_
```

*GPU time < 1 min; CPU fallback 2–3 min.*

---

## 7. Evaluation Protocol

| Metric                       | Definition / Usage                                                                                                             |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Silhouette score**   | Unsupervised cohesion / separation                                                                                             |
| **Davies–Bouldin**    | Lower = better cluster separation                                                                                              |
| **Purity**             | Match cluster → true label (labeled 5 k / 8 k)                                                                                |
| **NMI / ARI**          | Information & adjusted Rand indices                                                                                            |
| **Hungarian accuracy** | Map 10 clusters to 10 classes via Hungarian assignment → compute accuracy / F1 on**train-labeled**and**test**sets |
| **Confusion Matrix**   | Visualize mapping quality                                                                                                      |

Optionally add **k-NN (k=20)** on embeddings as a baseline; report top-1 accuracy.

---

## 8. Inference Pipeline

1. Input image
2. Same SimCLR augment **or** single center-crop → ResNet-50 → 2048-d feature
3. `pca.transform(feature)` → 100-d
4. `km.predict()` → cluster ID 0–9
5. Map cluster ID to semantic label via Hungarian mapping (stored from eval)

Per-image latency ≈ 5 ms (laptop GPU).

---

## 9. Timeline (8 Weeks)

| Week | Deliverable                                                                                    |
| ---- | ---------------------------------------------------------------------------------------------- |
| 1    | Environment setup, STL-10 loader, augmentation pipeline                                        |
| 2–3 | SimCLR training script, sanity check with linear probe                                         |
| 4    | Full 200-epoch pre-train, store checkpoints & logs                                             |
| 5    | Feature dump ➜ PCA ➜ initial K-means, quick metrics                                          |
| 6    | Hyper-parameter sweep (batch, τ, PCA dims), finalize centroids; compute all evaluation scores |
| 7    | Write report, produce slides; optional ablation (k-NN vs. K-means)                             |
| 8    | Buffer, polish code repo, final rehearsal                                                      |

---

## 10. Deliverables

* **Source code** (GitHub) with Docker/conda environment
* Trained **ResNet-50 weights** + **centroids.npy**
* Evaluation **report** (PDF) & presentation slides
* Log artifacts (W&B) for reproducibility

This plan gives a concrete, resource-aware path from raw STL-10 images to an unsupervised yet class-aligned label space, quantifying quality with both clustering and classification-style metrics.
