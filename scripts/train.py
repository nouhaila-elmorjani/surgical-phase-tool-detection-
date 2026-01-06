from typing import Tuple

import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from surgical_phase_tool.config import (
    TRAIN_MANIFEST,
    TEST_MANIFEST,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PHASE_LOSS_WEIGHT,
    TOOL_LOSS_WEIGHT,
    NUM_WORKERS,
    DEVICE,
    CHECKPOINT_PATH,
    SEED,
)
from surgical_phase_tool.config_loader import load_config, set_global_seed
from surgical_phase_tool.dataset import MultiTaskWindowDataset, PHASE_TO_ID, TOOL_COLUMNS
from surgical_phase_tool.models.resnet_multitask import PhaseToolNet
from surgical_phase_tool.hierarchy.phase_tool_mask import build_phase_tool_mask, apply_phase_mask_to_logits
from surgical_phase_tool.metrics import phase_metrics, tool_metrics


def compute_phase_class_weights(dataset: MultiTaskWindowDataset) -> torch.Tensor:
    counts = np.zeros(len(PHASE_TO_ID), dtype=np.int64)
    for _, phase_target, _ in DataLoader(dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS):
        ids = phase_target.argmax(dim=-1).numpy()
        for i in ids:
            counts[i] += 1
    counts = np.maximum(counts, 1)
    inv_freq = 1.0 / counts
    weights = inv_freq / inv_freq.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, phase_criterion, tool_criterion, phase_tool_mask):
    model.train()
    total_loss = 0.0

    for frames, phase_target, tool_target in loader:
        frames = frames.to(DEVICE)
        phase_target = phase_target.to(DEVICE)
        tool_target = tool_target.to(DEVICE)

        optimizer.zero_grad()

        phase_logits, tool_logits = model(frames)

                    
        phase_loss = phase_criterion(phase_logits, phase_target.argmax(dim=-1))

                                                              
        gt_phase_ids = phase_target.argmax(dim=-1)
        gt_phase_one_hot = torch.nn.functional.one_hot(gt_phase_ids, num_classes=len(PHASE_TO_ID)).float()
        masked_tool_logits = apply_phase_mask_to_logits(tool_logits, gt_phase_one_hot, phase_tool_mask, hard=True)

        tool_loss = tool_criterion(masked_tool_logits, tool_target)

        loss = PHASE_LOSS_WEIGHT * phase_loss + TOOL_LOSS_WEIGHT * tool_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * frames.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, phase_criterion, tool_criterion, phase_tool_mask) -> Tuple[dict, dict]:
    model.eval()

    all_phase_logits = []
    all_phase_targets = []
    all_tool_logits = []
    all_tool_targets = []

    total_loss = 0.0

    with torch.no_grad():
        for frames, phase_target, tool_target in loader:
            frames = frames.to(DEVICE)
            phase_target = phase_target.to(DEVICE)
            tool_target = tool_target.to(DEVICE)

            phase_logits, tool_logits = model(frames)

            phase_loss = phase_criterion(phase_logits, phase_target.argmax(dim=-1))

            gt_phase_ids = phase_target.argmax(dim=-1)
            gt_phase_one_hot = torch.nn.functional.one_hot(gt_phase_ids, num_classes=len(PHASE_TO_ID)).float()
            masked_tool_logits = apply_phase_mask_to_logits(tool_logits, gt_phase_one_hot, phase_tool_mask, hard=True)
            tool_loss = tool_criterion(masked_tool_logits, tool_target)

            loss = PHASE_LOSS_WEIGHT * phase_loss + TOOL_LOSS_WEIGHT * tool_loss

            total_loss += loss.item() * frames.size(0)

            all_phase_logits.append(phase_logits.cpu())
            all_phase_targets.append(phase_target.cpu())
            all_tool_logits.append(tool_logits.cpu())                                            
            all_tool_targets.append(tool_target.cpu())

    all_phase_logits = torch.cat(all_phase_logits, dim=0)
    all_phase_targets = torch.cat(all_phase_targets, dim=0)
    all_tool_logits = torch.cat(all_tool_logits, dim=0)
    all_tool_targets = torch.cat(all_tool_targets, dim=0)

    phase_overall_acc, phase_per_class_acc, cm = phase_metrics(all_phase_logits, all_phase_targets)

                                  
    tool_stats_no_mask = tool_metrics(all_tool_logits, all_tool_targets, threshold=0.5)

                                                                        
    phase_probs = all_phase_logits.softmax(dim=-1)
    phase_tool_mask = phase_tool_mask.to(all_tool_logits.device)
    masked_logits_pred = apply_phase_mask_to_logits(all_tool_logits, phase_probs, phase_tool_mask, hard=False)
    tool_stats_with_mask = tool_metrics(masked_logits_pred, all_tool_targets, threshold=0.5)

    eval_stats = {
        "loss": total_loss / len(loader.dataset),
        "phase_overall_acc": phase_overall_acc,
        "phase_per_class_acc": phase_per_class_acc,
        "confusion_matrix": cm,
        "tool_no_mask": tool_stats_no_mask,
        "tool_with_mask": tool_stats_with_mask,
    }

    return eval_stats


def main():
    """Train PhaseToolNet and evaluate on the test split.

    Logs epoch-level metrics to CSV and saves the best checkpoint.
    """

    load_config()
    set_global_seed(SEED)

    print(f"Using device: {DEVICE}")

    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    train_log_path = os.path.join(logs_dir, "training_log.csv")

    full_train_dataset = MultiTaskWindowDataset(TRAIN_MANIFEST, is_train=True)

    phase_class_weights = compute_phase_class_weights(full_train_dataset).to(DEVICE)
    print("Phase class weights:", phase_class_weights.cpu().numpy())

    val_size = max(1, int(0.1 * len(full_train_dataset)))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    test_dataset = MultiTaskWindowDataset(TEST_MANIFEST, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    model = PhaseToolNet(backbone_name="resnet18", pretrained=True).to(DEVICE)

    phase_criterion = nn.CrossEntropyLoss(weight=phase_class_weights)
    tool_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    phase_tool_mask = build_phase_tool_mask().to(DEVICE)

    best_val_acc = 0.0
    best_state = None
    patience = 5
    epochs_no_improve = 0

    with open(train_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_phase_acc",
                "val_tool_micro_no_mask",
                "val_tool_micro_with_mask",
            ]
        )

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                phase_criterion,
                tool_criterion,
                phase_tool_mask,
            )

            val_stats = evaluate(model, val_loader, phase_criterion, tool_criterion, phase_tool_mask)

            scheduler.step(val_stats["phase_overall_acc"])

            print(
                f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_stats['loss']:.4f} | Val Phase Acc: {val_stats['phase_overall_acc']:.4f}"
            )
            print(
                "  Val Tool Micro-F1 (no mask / with mask): "
                f"{val_stats['tool_no_mask']['f1_micro']:.4f} / {val_stats['tool_with_mask']['f1_micro']:.4f}"
            )

            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_stats['loss']:.6f}",
                    f"{val_stats['phase_overall_acc']:.6f}",
                    f"{val_stats['tool_no_mask']['f1_micro']:.6f}",
                    f"{val_stats['tool_with_mask']['f1_micro']:.6f}",
                ]
            )

            if val_stats["phase_overall_acc"] > best_val_acc:
                best_val_acc = val_stats["phase_overall_acc"]
                best_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        try:
            torch.save(best_state, CHECKPOINT_PATH)
            print(f"Saved best model checkpoint to {CHECKPOINT_PATH}")
        except OSError as exc:
            print(f"Warning: could not save checkpoint: {exc}")

    print("\n=== FINAL TEST EVALUATION ===")
    test_stats = evaluate(model, test_loader, phase_criterion, tool_criterion, phase_tool_mask)

    print(f"Test Loss: {test_stats['loss']:.4f}")
    print(f"Overall Phase Accuracy: {test_stats['phase_overall_acc']:.4f}")
    print("Per-class Phase Accuracy (order of PHASE_TO_ID):")
    for phase_name, phase_id in PHASE_TO_ID.items():
        acc = test_stats["phase_per_class_acc"][phase_id]
        print(f"  {phase_id} - {phase_name}: {acc:.4f}")

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(test_stats["confusion_matrix"])

    print("\n=== TOOL METRICS WITHOUT PHASE MASKING ===")
    no_mask = test_stats["tool_no_mask"]
    print(
        f"Micro-F1: {no_mask['f1_micro']:.4f} "
        f"(P={no_mask['precision_micro']:.4f}, R={no_mask['recall_micro']:.4f})"
    )
    for i, tool in enumerate(TOOL_COLUMNS):
        print(
            f"  {tool}: P={no_mask['precision_per'][i]:.4f}, "
            f"R={no_mask['recall_per'][i]:.4f}, F1={no_mask['f1_per'][i]:.4f}"
        )

    print("\n=== TOOL METRICS WITH HIERARCHICAL PHASE MASKING ===")
    with_mask = test_stats["tool_with_mask"]
    print(
        f"Micro-F1: {with_mask['f1_micro']:.4f} "
        f"(P={with_mask['precision_micro']:.4f}, R={with_mask['recall_micro']:.4f})"
    )
    for i, tool in enumerate(TOOL_COLUMNS):
        print(
            f"  {tool}: P={with_mask['precision_per'][i]:.4f}, "
            f"R={with_mask['recall_per'][i]:.4f}, F1={with_mask['f1_per'][i]:.4f}"
        )

    print("\nHierarchy effect (Micro-F1):")
    print(f"  Without mask: {no_mask['f1_micro']:.4f}")
    print(f"  With mask   : {with_mask['f1_micro']:.4f}")


if __name__ == "__main__":
    main()
