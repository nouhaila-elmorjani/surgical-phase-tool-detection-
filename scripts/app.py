"""Streamlit demo for surgical phase and tool recognition.

"""

from __future__ import annotations

import io
from typing import List, Tuple

import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

from surgical_phase_tool.config import IMAGE_SIZE, WINDOW_SIZE, DEVICE, CHECKPOINT_PATH, SEED
from surgical_phase_tool.config_loader import load_config, set_global_seed
from surgical_phase_tool.dataset import PHASE_TO_ID, TOOL_COLUMNS
from surgical_phase_tool.hierarchy.phase_tool_mask import build_phase_tool_mask, apply_phase_mask_to_logits
from surgical_phase_tool.models.resnet_multitask import PhaseToolNet


@st.cache_resource
def load_model() -> Tuple[PhaseToolNet, torch.Tensor, T.Compose, torch.device]:
    """Load model, hierarchy mask and preprocessing pipeline."""

    load_config()
    set_global_seed(SEED)

    device = torch.device(DEVICE)

    model = PhaseToolNet(backbone_name="resnet18", pretrained=False).to(device)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    phase_tool_mask = build_phase_tool_mask().to(device)

    transforms = T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return model, phase_tool_mask, transforms, device


def preprocess_image(file_bytes: bytes, transforms: T.Compose) -> Tuple[Image.Image, torch.Tensor]:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = transforms(img)
    return img, tensor


def run_inference(
    model: PhaseToolNet,
    mask: torch.Tensor,
    frames_tensor: torch.Tensor,
    device: torch.device,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Run model and return phase id and probabilities plus tool probs.

    Returns
    -------
    phase_id : int
        Index of the predicted phase.
    phase_probs : np.ndarray
        Probability for each phase.
    tool_probs : np.ndarray
        Tool probabilities before hierarchy masking.
    masked_probs : np.ndarray
        Tool probabilities after hierarchy masking.
    """

    with torch.no_grad():
        phase_logits, tool_logits = model(frames_tensor.to(device))
        phase_probs_t = phase_logits.softmax(dim=-1)
        phase_id = int(phase_probs_t.argmax(dim=-1)[0].item())

        phase_probs = phase_probs_t[0].cpu().numpy()
        tool_probs = tool_logits.sigmoid()[0].cpu().numpy()
        masked_logits = apply_phase_mask_to_logits(tool_logits, phase_probs_t, mask, hard=False)
        masked_probs = masked_logits.sigmoid()[0].cpu().numpy()

    return phase_id, phase_probs, tool_probs, masked_probs


def main() -> None:
    st.title("Surgical Phase and Tool Recognition")

    st.markdown(
        "Upload one or more laparoscopic frames. Each frame is repeated to "
        "form a short temporal window for the PhaseToolNet model."
    )

    model, phase_tool_mask, transforms, device = load_model()

    uploaded_files = st.file_uploader(
        "Upload frames (PNG/JPEG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Upload at least one frame to run inference.")
        return

    phase_names = [name for name, _ in sorted(PHASE_TO_ID.items(), key=lambda kv: kv[1])]

    timeline_phases: List[int] = []

    for idx, uploaded in enumerate(uploaded_files):
        bytes_data = uploaded.read()
        img, tensor = preprocess_image(bytes_data, transforms)

        window = torch.stack([tensor] * WINDOW_SIZE, dim=0).unsqueeze(0)

        phase_id, phase_probs, tool_probs, masked_probs = run_inference(
            model, phase_tool_mask, window, device
        )
        timeline_phases.append(phase_id)

        phase_name = phase_names[phase_id]

        st.subheader(f"Frame {idx}: predicted phase = {phase_name}")

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption=f"Frame {idx}")

        with col2:
            st.markdown("**Phase probabilities**")
            phase_table = {
                "phase": phase_names,
                "probability": phase_probs,
            }
            st.dataframe(phase_table)

            st.markdown("**Tool probabilities (unmasked vs masked)**")
            tool_table = {
                "tool": TOOL_COLUMNS,
                "prob_unmasked": tool_probs,
                "prob_masked": masked_probs,
            }
            st.dataframe(tool_table)

        st.markdown("**Hierarchy mask row for predicted phase**")
        mask_row = phase_tool_mask[phase_id].cpu().numpy()
        mask_table = {"tool": TOOL_COLUMNS, "allowed": mask_row.astype(int)}
        st.dataframe(mask_table)

        st.markdown("---")

    if len(timeline_phases) > 1:
        import pandas as pd

        df = pd.DataFrame(
            {
                "frame_index": np.arange(len(timeline_phases)),
                "phase_id": timeline_phases,
                "phase": [phase_names[p] for p in timeline_phases],
            }
        )
        st.subheader("Phase timeline across uploaded frames")
        st.line_chart(df.set_index("frame_index")["phase_id"])


if __name__ == "__main__":
    main()
