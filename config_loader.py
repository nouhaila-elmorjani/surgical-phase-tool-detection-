import os
import yaml


def _resolve_device(cfg: dict) -> str:
    

    runtime = cfg.get("runtime", {})
    device_cfg = runtime.get("device", "cpu")

    if device_cfg == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    return device_cfg


def load_config(config_path: str = None) -> dict:
    """Load YAML configuration for the project.

    This is the single source of truth for paths, hyperparameters,
    and runtime options. Defaults to configs/final_config.yaml next
    to this module.
    """

    if config_path is None:
        here = os.path.dirname(__file__)
        config_path = os.path.join(here, "configs", "final_config.yaml")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve and normalize a few commonly used paths/values
    data_root = os.path.expanduser(cfg["data_root"])
    cfg["data_root_resolved"] = data_root

    # Derived paths for current split
    split = cfg.get("split", "split_2")
    processed_root = os.path.join(data_root, cfg["paths"]["processed_root"])
    multi_task_root = os.path.join(data_root, cfg["paths"]["splits_multi_task"], split)

    cfg["paths"]["image_root_resolved"] = processed_root
    cfg["paths"]["train_manifest_resolved"] = os.path.join(
        multi_task_root, cfg["paths"]["train_manifest"]
    )
    cfg["paths"]["test_manifest_resolved"] = os.path.join(
        multi_task_root, cfg["paths"]["test_manifest"]
    )

    # Checkpoint path is stored relative to package root
    cfg["checkpoint_resolved"] = os.path.join(
        os.path.dirname(__file__), cfg["checkpoint"]["path"]
    )

    # Resolve device (CPU / CUDA / auto)
    cfg.setdefault("runtime", {})["device_resolved"] = _resolve_device(cfg)

    return cfg


def get_device(cfg: dict) -> str:
    """Return the resolved device string (e.g. ``"cpu"`` or ``"cuda"``)."""

    return cfg["runtime"].get("device_resolved", "cpu")


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility on CPU-only runs."""

    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Ensure deterministic behavior where possible
    torch.use_deterministic_algorithms(True, warn_only=True)
