# Surgical Phase and Tool Recognition

This repository implements a multitask deep learning model for laparoscopic surgery video analysis. The model jointly predicts the surgical **phase** and the presence of key **instruments/tools** in short clips, and optionally applies a phase→tool hierarchy to enforce clinically plausible tool predictions.

The codebase includes:

- A ResNet-based multitask architecture (`PhaseToolNet`) with phase-conditional tool prediction
- Hierarchy-aware masking of tool logits based on the predicted phase
- Reproducible training and evaluation pipelines
- A small CLI demo for running inference on a folder of frames
- A Streamlit app for interactive inspection of model predictions

---

## Repository Structure

- [app.py](app.py) – Streamlit app for interactive phase/tool recognition on single frames
- [train.py](train.py) – Training script for `PhaseToolNet` (train/val/test split handling, logging, checkpointing)
- [evaluate.py](evaluate.py) – Test evaluation from a saved checkpoint with CSV export of metrics
- [demo.py](demo.py) – Simple CLI demo that runs inference on a folder of frames
- [config_loader.py](config_loader.py) – YAML configuration loader and device/seed utilities
- [config.py](config.py) – Backwards-compatible config shim exposing constants from the YAML config
- [dataset.py](dataset.py) – `MultiTaskWindowDataset` for windowed video clips and label loading
- [hierarchy.py](hierarchy.py) / [hierarchy/phase_tool_mask.py](hierarchy/phase_tool_mask.py) – Phase→tool validity mask and masking utilities
- [metrics.py](metrics.py) – Phase and tool metrics (accuracy, confusion matrix, per-tool precision/recall/F1)
- [models/resnet_multitask.py](models/resnet_multitask.py) – Multitask ResNet backbone and heads
- [utils/report_utils.py](utils/report_utils.py) – Helpers for exporting figures and tables from notebooks/scripts
- [configs/final_config.yaml](configs/final_config.yaml) – Single source of truth for paths, hyperparameters, and runtime options
- [requirements.txt](requirements.txt) – Python dependencies

Notebooks in [notebooks/](notebooks/) illustrate dataset inspection, model behavior, training curves, and final evaluation.

---

## Installation

1. **Clone the repository**

   ```bash
   git clone <YOUR_REPO_URL>
   cd source_codes
   ```

2. **Create and activate a Python environment** (recommended)

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on macOS / Linux
   # .venv\\Scripts\\activate  # on Windows
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

The code targets a recent PyTorch and torchvision (see versions in `requirements.txt`).

---

## Dataset and Configuration

All paths and hyperparameters are configured via [configs/final_config.yaml](configs/final_config.yaml) and loaded through [config_loader.py](config_loader.py).

Key entries:

- `data_root`: root directory containing the processed dataset
- `paths.processed_root`: path (relative to `data_root`) with processed frames
- `paths.splits_multi_task`: directory (relative to `data_root`) containing the multitask train/test splits
- `paths.train_manifest` / `paths.test_manifest`: CSV manifests for training and test sets
- `model.*`: number of phase and tool classes
- `training.*`: image size, temporal window length, batch size, learning rate, etc.
- `runtime.device`: device selection (`"cpu"`, `"cuda"`, or `"auto"`)
- `checkpoint.path`: relative path (from the project root) where the best model checkpoint is saved/loaded

By default, `data_root` points to a folder under your Desktop. Update this field to match the location of your processed dataset before running training or evaluation.

The helper [config.py](config.py) exposes commonly used configuration values as module-level constants for older scripts, but new code should prefer the dict returned by `config_loader.load_config()`.

---

## Training

The main training entry point is [train.py](train.py). It:

- Loads configuration and sets global seeds
- Builds `MultiTaskWindowDataset` for train/validation/test
- Computes class weights for the phase loss
- Trains `PhaseToolNet` with a weighted combination of phase and tool losses
- Applies the phase→tool hierarchy during tool loss computation
- Logs epoch-level metrics to `logs/training_log.csv`
- Saves the best-performing checkpoint (by validation phase accuracy) to the location defined in the config

Run training:

```bash
python train.py
```

Training progress and metrics (losses, accuracies, tool F1 scores with and without masking) are printed to stdout and logged to CSV.

---

## Evaluation

Use [evaluate.py](evaluate.py) to evaluate a trained checkpoint on the test split:

```bash
python evaluate.py
```

This script:

- Loads the saved checkpoint specified in [configs/final_config.yaml](configs/final_config.yaml)
- Computes phase accuracy (overall and per-class) and a confusion matrix
- Computes tool metrics with and without hierarchy-based masking
- Prints a detailed summary to stdout
- Writes:
  - `logs/eval_summary.csv` – high-level metrics
  - `logs/eval_tools.csv` – per-tool precision/recall/F1 with and without masking

---

## CLI Demo

The lightweight demo in [demo.py](demo.py) runs inference on a folder of still frames:

```bash
python demo.py \
  --input_frames path/to/frames_folder \
  --checkpoint path/to/checkpoint.pth
```

It prints the predicted phase and the top tools before and after applying the hierarchy-based mask.

---

## Streamlit App

[app.py](app.py) provides an interactive Streamlit UI for inspecting predictions frame-by-frame.

Start the app with:

```bash
streamlit run app.py
```

The app lets you:

- Upload one or more single frames (PNG/JPEG)
- View predicted phases, full phase probability distributions, and tool probabilities
- Compare raw vs. hierarchy-masked tool probabilities
- Inspect the hierarchy mask row corresponding to the predicted phase
- Visualize the predicted phase timeline across multiple uploaded frames

Ensure that the checkpoint path in [configs/final_config.yaml](configs/final_config.yaml) points to a valid trained model before running the app.

---

## Reproducibility

Reproducibility is handled via:

- `training.seed` in [configs/final_config.yaml](configs/final_config.yaml)
- `set_global_seed()` in [config_loader.py](config_loader.py), which seeds Python, NumPy, and PyTorch and enables deterministic algorithms where possible

For strictly reproducible experiments on GPU, you may need to additionally configure CUDA/cuDNN determinism flags according to your hardware and PyTorch version.

---
