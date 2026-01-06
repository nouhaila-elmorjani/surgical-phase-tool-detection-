# 🩺 Surgical Phase & Tool Recognition

A **multitask deep learning framework** for laparoscopic surgery video analysis, jointly recognizing **surgical phases** and **surgical tools** from short temporal windows of frames.
The system integrates **clinical workflow constraints** through a **phase → tool hierarchy**, improving robustness, interpretability, and practical usability.

---

## ✨ Highlights

* Joint **surgical phase recognition** and **tool detection**
* Clinically informed **phase-conditioned tool predictions**
* Temporal window–based video modeling
* Fully reproducible training and evaluation pipeline
* Command-line inference and interactive visualization
* Centralized YAML-based configuration

Designed for **research, benchmarking, and experimental deployment** in surgical workflow understanding.

---

## 🧠 Conceptual Overview

Laparoscopic procedures follow a well-defined sequence of surgical phases, and only a subset of instruments is clinically plausible within each phase.
This project exploits that structure by:

1. Predicting the **current surgical phase**
2. Using the predicted phase to **constrain tool predictions**
3. Producing **clinically consistent and interpretable outputs**

This design reduces implausible predictions and improves tool detection reliability.

---

## 🏗️ Repository Structure

```
.
├── app.py                     # Streamlit interface for interactive inspection
├── train.py                   # Training pipeline
├── evaluate.py                # Test evaluation and metric export
├── demo.py                    # CLI inference on frame folders
├── config_loader.py           # YAML loader, seed and device utilities
├── config.py                  # Backward-compatible config access
├── dataset.py                 # Temporal multitask dataset
├── hierarchy/
│   └── phase_tool_mask.py     # Phase → tool validity definitions
├── metrics.py                 # Phase and tool evaluation metrics
├── models/
│   └── resnet_multitask.py    # Multitask ResNet architecture
├── utils/
│   └── report_utils.py        # Reporting helpers
├── configs/
│   └── final_config.yaml      # Central configuration file
├── notebooks/                 # Analysis and visualization notebooks
└── requirements.txt           # Dependencies
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/nouhaila-elmorjani/surgical-phase-tool-detection
cd surgical-phase-tool-detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🗂️ Dataset & Configuration

All paths, hyperparameters, and runtime options are defined in:

```
configs/final_config.yaml
```

### Key configuration fields

* **Data**

  * `data_root`: Root directory of the processed dataset
  * `paths.processed_root`: Directory containing extracted frames
  * `paths.splits_multi_task`: Train/validation/test splits
  * `paths.train_manifest`, `paths.test_manifest`: CSV manifests

* **Model**

  * `model.num_phases`
  * `model.num_tools`

* **Training**

  * `training.window_size`
  * `training.image_size`
  * `training.batch_size`
  * `training.learning_rate`
  * `training.seed`

* **Runtime**

  * `runtime.device`: `"cpu"`, `"cuda"`, or `"auto"`

* **Checkpoint**

  * `checkpoint.path`: Location of the saved model

⚠️ Update `data_root` before running training or evaluation.

---

## 🚀 Training

Train the model from scratch:

```bash
python train.py
```

The training pipeline:

* Sets global random seeds
* Builds temporal window datasets
* Computes class weights to address phase imbalance
* Optimizes a weighted multitask objective:

  * Phase classification loss
  * Tool detection loss with hierarchy masking
* Logs metrics to:

  ```
  logs/training_log.csv
  ```
* Saves the best checkpoint based on **validation phase accuracy**

---

## 📊 Evaluation

Evaluate a trained model on the test set:

```bash
python evaluate.py
```

This script computes:

* Overall and per-class phase accuracy
* Phase confusion matrix
* Tool precision, recall, and F1 scores
* Tool metrics **with and without hierarchy masking**

Results are written to:

```
logs/eval_summary.csv
logs/eval_tools.csv
```

---

## 🧪 CLI Inference Demo

Run inference on a directory of frames:

```bash
python demo.py \
  --input_frames path/to/frames_folder \
  --checkpoint path/to/checkpoint.pth
```

---

## 🖥️ Interactive Streamlit App

Launch the interactive interface:

```bash
streamlit run app.py
```

The app allows you to:

* Upload one or more frames
* Inspect phase predictions and probability distributions
* Compare raw vs hierarchy-masked tool predictions
* Visualize phase evolution across frames
* Inspect the active phase → tool validity mask

---

## 🧩 Model & Hierarchy Design

### Architecture

* Shared **ResNet** backbone
* Two task-specific heads:

  * **Phase head** (softmax)
  * **Tool head** (sigmoid)

### Phase → Tool Hierarchy

Clinical constraints are encoded as a **binary validity matrix** defining which tools are plausible in each surgical phase.

The hierarchy is applied:

* **During training**: invalid tools are excluded from the loss
* **During inference**: tool probabilities are filtered based on the predicted phase

---

## 🔁 Reproducibility

Reproducibility is supported through:

* Fixed random seeds (Python, NumPy, PyTorch)
* Centralized YAML configuration

---

## 👤 Author

**ELMORJANI Nouhaila**
GitHub: [https://github.com/nouhaila-elmorjani](https://github.com/nouhaila-elmorjani)

---

## 🙏 Acknowledgments

This work builds upon and is inspired by prior research and open resources in surgical workflow analysis, including:

* **Surgformer: Surgical Transformer with Hierarchical Temporal Attention for Surgical Phase Recognition**
  Yang, Shu; Luo, Luyang; Wang, Qiong; Chen, Hao
  MICCAI 2024 (Open Access)
  [https://papers.miccai.org/miccai-2024/paper/1220_paper.pdf](https://papers.miccai.org/miccai-2024/paper/1220_paper.pdf)
  [https://github.com/isyangshu/Surgformer](https://github.com/isyangshu/Surgformer)

* **PhaKIR Dataset – Surgical Phase, Keypoint, and Instrument Recognition**
  Tobias Rueckert *et al.*
  MICCAI 2024 / EndoVis Challenge
  A multi-institutional dataset providing frame-level annotations for surgical phase and instrument recognition, enabling research on temporally consistent surgical scene understanding.

We thank the authors and organizers for making these resources available to the research community.

---

