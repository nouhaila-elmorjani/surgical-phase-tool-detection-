from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from config import NUM_PHASE_CLASSES, NUM_TOOL_CLASSES


def phase_metrics(phase_logits: torch.Tensor, phase_targets: torch.Tensor) -> Tuple[float, np.ndarray, np.ndarray]:
    with torch.no_grad():
        preds = phase_logits.softmax(dim=-1).argmax(dim=-1).cpu().numpy()
        targets = phase_targets.argmax(dim=-1).cpu().numpy()

    overall_acc = (preds == targets).mean()

    cm = confusion_matrix(targets, preds, labels=list(range(NUM_PHASE_CLASSES)))

    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    return overall_acc, per_class_acc, cm


def tool_metrics(tool_logits: torch.Tensor, tool_targets: torch.Tensor, threshold: float = 0.5):
    with torch.no_grad():
        probs = tool_logits.sigmoid().cpu().numpy()
        preds = (probs >= threshold).astype(np.int32)
        targets = tool_targets.cpu().numpy().astype(np.int32)

    # micro-averaged metrics
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        targets.reshape(-1), preds.reshape(-1), average="micro", zero_division=0
    )

    # per-tool metrics
    precision_per, recall_per, f1_per, _ = precision_recall_fscore_support(
        targets, preds, average=None, zero_division=0
    )

    return {
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_per": precision_per,
        "recall_per": recall_per,
        "f1_per": f1_per,
    }
