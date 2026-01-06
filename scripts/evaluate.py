import csv
import os

import torch

from surgical_phase_tool.config import (
    TRAIN_MANIFEST,
    TEST_MANIFEST,
    BATCH_SIZE,
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


def evaluate_model():
    """Run test evaluation using the saved checkpoint.

    Prints metrics and writes CSV logs for reproducibility.
    """

    load_config()
    set_global_seed(SEED)

    print(f"Using device: {DEVICE}")

    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    summary_path = os.path.join(logs_dir, "eval_summary.csv")
    tools_path = os.path.join(logs_dir, "eval_tools.csv")

    test_dataset = MultiTaskWindowDataset(TEST_MANIFEST, is_train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    model = PhaseToolNet(backbone_name="resnet18", pretrained=False)
    model.to(DEVICE)

    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state)

    phase_tool_mask = build_phase_tool_mask().to(DEVICE)

    model.eval()

    all_phase_logits = []
    all_phase_targets = []
    all_tool_logits = []
    all_tool_targets = []

    with torch.no_grad():
        for frames, phase_target, tool_target in test_loader:
            frames = frames.to(DEVICE)
            phase_target = phase_target.to(DEVICE)
            tool_target = tool_target.to(DEVICE)

            phase_logits, tool_logits = model(frames)

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

    masked_logits = apply_phase_mask_to_logits(
        all_tool_logits, phase_probs, phase_tool_mask, hard=False
    )
    tool_stats_with_mask = tool_metrics(masked_logits, all_tool_targets, threshold=0.5)

    print("\n=== TEST EVALUATION ===")
    print(f"Overall Phase Accuracy: {phase_overall_acc:.4f}")
    print("Per-class Phase Accuracy (order of PHASE_TO_ID):")
    for phase_name, phase_id in PHASE_TO_ID.items():
        acc = phase_per_class_acc[phase_id]
        print(f"  {phase_id} - {phase_name}: {acc:.4f}")

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    print("\n=== TOOL METRICS WITHOUT PHASE MASKING ===")
    no_mask = tool_stats_no_mask
    print(
        f"Micro-F1: {no_mask['f1_micro']:.4f} "
        f"(P={no_mask['precision_micro']:.4f}, R={no_mask['recall_micro']:.4f})"
    )
    for i, tool in enumerate(TOOL_COLUMNS):
        print(
            f"  {tool}: P={no_mask['precision_per'][i]:.4f}, R={no_mask['recall_per'][i]:.4f}, F1={no_mask['f1_per'][i]:.4f}"
        )

    print("\n=== TOOL METRICS WITH HIERARCHICAL PHASE MASKING ===")
    with_mask = tool_stats_with_mask
    print(
        f"Micro-F1: {with_mask['f1_micro']:.4f} "
        f"(P={with_mask['precision_micro']:.4f}, R={with_mask['recall_micro']:.4f})"
    )
    for i, tool in enumerate(TOOL_COLUMNS):
        print(
            f"  {tool}: P={with_mask['precision_per'][i]:.4f}, R={with_mask['recall_per'][i]:.4f}, F1={with_mask['f1_per'][i]:.4f}"
        )

    print("\nHierarchy effect (Micro-F1):")
    print(f"  Without mask: {no_mask['f1_micro']:.4f}")
    print(f"  With mask   : {with_mask['f1_micro']:.4f}")

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "phase_overall_acc",
                "tool_micro_f1_no_mask",
                "tool_micro_f1_with_mask",
            ]
        )
        writer.writerow(
            [
                f"{phase_overall_acc:.6f}",
                f"{no_mask['f1_micro']:.6f}",
                f"{with_mask['f1_micro']:.6f}",
            ]
        )

    with open(tools_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tool",
                "precision_no_mask",
                "recall_no_mask",
                "f1_no_mask",
                "precision_with_mask",
                "recall_with_mask",
                "f1_with_mask",
            ]
        )
        for i, tool in enumerate(TOOL_COLUMNS):
            writer.writerow(
                [
                    tool,
                    f"{no_mask['precision_per'][i]:.6f}",
                    f"{no_mask['recall_per'][i]:.6f}",
                    f"{no_mask['f1_per'][i]:.6f}",
                    f"{with_mask['precision_per'][i]:.6f}",
                    f"{with_mask['recall_per'][i]:.6f}",
                    f"{with_mask['f1_per'][i]:.6f}",
                ]
            )


if __name__ == "__main__":
    evaluate_model()
