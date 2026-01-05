"""Backward-compatible config shim.

The project now uses a single YAML configuration file loaded via
config_loader.load_config. This module exposes the same constants as
before so existing imports keep working, but new code should
prefer using the dict returned by load_config.
"""

from config_loader import load_config, get_device


_CFG = load_config()

DATA_ROOT = _CFG["data_root_resolved"]
MULTI_TASK_SPLIT_DIR = _CFG["paths"]["train_manifest_resolved"].rsplit("/", 1)[0]
TRAIN_MANIFEST = _CFG["paths"]["train_manifest_resolved"]
TEST_MANIFEST = _CFG["paths"]["test_manifest_resolved"]

IMAGE_ROOT = _CFG["paths"]["image_root_resolved"]

NUM_PHASE_CLASSES = _CFG["model"]["num_phases"]
NUM_TOOL_CLASSES = _CFG["model"]["num_tools"]

IMAGE_SIZE = _CFG["training"]["image_size"]
WINDOW_SIZE = _CFG["training"]["window_size"]

BATCH_SIZE = _CFG["training"]["batch_size"]
NUM_EPOCHS = _CFG["training"]["num_epochs"]
LEARNING_RATE = _CFG["training"]["learning_rate"]
WEIGHT_DECAY = _CFG["training"]["weight_decay"]

PHASE_LOSS_WEIGHT = _CFG["training"]["phase_loss_weight"]
TOOL_LOSS_WEIGHT = _CFG["training"]["tool_loss_weight"]

NUM_WORKERS = _CFG["training"]["num_workers"]

DEVICE = get_device(_CFG)

CHECKPOINT_PATH = _CFG["checkpoint_resolved"]

SEED = _CFG["training"].get("seed", 42)
