# Projet-AIfTWD: Trustworthy Federated Learning for Cancer Detection

Implementation of a Trust-worthy Federated Learning setup for Breast Cancer Identification using the **CBIS-DDSM** dataset. This project explores the trade-offs between **Performance**, **Robustness** (Non-IID data), **Privacy** (Differential Privacy), and **Explainability**.

## 📂 1. Dataset Setup

The dataset images must be downloaded and placed in the folder `Partie2/dataset/` (or wherever your script points to).

* **Source:** [Kaggle CBIS-DDSM Dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset?resource=download)
* **Preparation:** Run the preparation script to resize images and generate the clean CSV.

```bash
# Prepare images (Resize to 224x224 & Indexing)
python3 Partie2/prepare_dataset.py

```

---

## 🚀 2. Experiments & Commands

All experiments use `main.py`. Ensure your `client_resources` are set to `{'num_cpus': 1, 'num_gpus': 1.0}` to avoid OOM errors on GPUs like P100/T4.

```bash
pip install -q opacus flwr
```

### 🥇 Phase 1: The "Gold Standard" (Centralized)

**Goal:** Establish the maximum theoretical performance (Upper Bound) by training on all data without federation.

```bash
python3 Partie2/main.py \
  --mode centralized \
  --train_id centralized_resnet50 \
  --model resnet50 \
  --batch_size 32 \
  --epochs 50 \
  --lr 0.001

```

### 🛡️ Phase 2: Robustness (FedAvg vs. FedProx)

**Goal:** Compare performance on **Non-IID data** (Label Skew).

* *Client 0:* Mostly Malignant cases.
* *Client 1:* Mostly Benign cases (Screening center).

**A. Baseline (FedAvg):** Should struggle with heterogeneity.

```bash
python3 Partie2/main.py \
  --mode federated \
  --train_id fedavg_noniid_baseline \
  --algo fedavg \
  --model resnet50 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.001

```

**B. Robust Solution (FedProx):** Should stabilize convergence using the proximal term ().

```bash
python3 Partie2/main.py \
  --mode federated \
  --train_id fedprox_noniid_robust \
  --algo fedprox \
  --mu 0.1 \
  --model resnet50 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.001

```

### 🔒 Phase 3: Privacy (Differential Privacy)

**Goal:** Analyze the **Privacy-Utility Trade-off**. We add noise () and clip gradients () to guarantee mathematically proven anonymity.

*Note: Batch size is reduced to 16 to handle the memory overhead of Opacus.*

```bash
python3 Partie2/main.py \
  --mode federated \
  --train_id dp_resnet50_secure \
  --algo fedavg \
  --dp \
  --dp_noise 1.0 \
  --dp_clip 1.2 \
  --model resnet50 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.001

```

---

## 🧠 3. Explainability & Evaluation

Once the models are trained and saved in the `models/` folder, use the provided notebook cells or analysis scripts to generate:

1. **Grad-CAM Heatmaps:** To visualize if the model focuses on the tumor or the background.
2. **Confusion Matrices:** To check for bias (e.g., False Negatives on Malignant cases).

**Example Code for Grad-CAM:**

```python
from Partie2.src.models import get_model
import torch

# Load best model
model = get_model(model_name='resnet50', num_classes=4, device='cuda')
model.load_state_dict(torch.load("models/fedprox_noniid_robust_latest.pth"))

# Run Grad-CAM visualization function (see notebook)
plot_gradcam(model, test_dataset)

```

---

## 🛠️ Troubleshooting

* **CUDA Out of Memory (OOM):**
* Reduce `--batch_size` to 16 or 8.
* Ensure `client_resources={"num_cpus": 1, "num_gpus": 1.0}` is set in `main.py` (do not use 0.5 GPUs).
* Restart the kernel to clear phantom memory.


* **"Inplace" Error (RuntimeError):**
* Ensure `src/models.py` includes the "Monkey Patch" for ResNet BasicBlock/Bottleneck.


* **Metrics connection failed (RPC code 14):**
* Ignore this error on Kaggle/Colab. It's just the Ray dashboard failing to display stats; the training continues in the background.



python Partie2/main.py --mode centralized --train_id centralized_resnet50 --model resnet50 --batch_size 32 --epochs 50 --lr 0.001