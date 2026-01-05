import torch

from dataset import PHASE_TO_ID, TOOL_COLUMNS


# Define which tools are valid for each phase (binary mask)
# Order of tools = TOOL_COLUMNS

PHASE_TOOL_VALIDITY = {
    "Preparation": [
        0,  # Argonbeamer
        0,  # Clip-Applicator
        0,  # Drainage
        0,  # Grasper
        0,  # HF-Coagulation-Probe
        0,  # Needle-Probe
        0,  # Palpation-Probe
        1,  # PE-Forceps
        0,  # Scissor
        0,  # Suction-Rod
        0,  # Trocar-Tip
    ],
    "CalotTriangleDissection": [
        0,
        1,  # Clip-Applicator
        0,
        0,  # Grasper (not used in your main labels)
        0,
        0,
        0,
        1,  # PE-Forceps
        1,  # Scissor
        0,
        0,
    ],
    "ClippingCutting": [
        0,
        1,  # Clip-Applicator
        0,
        0,
        0,
        0,
        0,
        0,
        1,  # Scissor
        0,
        0,
    ],
    "GallbladderDissection": [
        0,
        0,
        0,
        0,
        1,  # HF-Coagulation-Probe
        0,
        0,
        1,  # PE-Forceps
        1,  # Scissor
        1,  # Suction-Rod
        0,
    ],
    "CleaningCoagulation": [
        0,
        0,
        0,
        0,
        1,  # HF-Coagulation-Probe
        0,
        0,
        0,
        0,
        1,  # Suction-Rod
        0,
    ],
    "GallbladderPackaging": [
        0,
        0,
        0,
        0,
        0,
        1,  # Needle-Probe
        0,
        1,  # PE-Forceps
        0,
        0,
        0,
    ],
    "GallbladderRetraction": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,  # PE-Forceps
        0,
        0,
        1,  # Trocar-Tip
    ],
    "Undefined": [
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0
    ],
}


def build_phase_tool_mask() -> torch.Tensor:
    num_phases = len(PHASE_TO_ID)
    num_tools = len(TOOL_COLUMNS)
    mask = torch.zeros(num_phases, num_tools, dtype=torch.float32)
    for phase_name, phase_id in PHASE_TO_ID.items():
        valid_list = PHASE_TOOL_VALIDITY[phase_name]
        if len(valid_list) != num_tools:
            raise ValueError(f"Validity list for {phase_name} has wrong length {len(valid_list)}")
        mask[phase_id] = torch.tensor(valid_list, dtype=torch.float32)
    return mask  # (num_phases, num_tools)


def apply_phase_mask_to_logits(tool_logits: torch.Tensor, phase_probs: torch.Tensor, mask: torch.Tensor, hard: bool = False) -> torch.Tensor:
    # tool_logits: (B, num_tools)
    # phase_probs: (B, num_phases)
    # mask: (num_phases, num_tools)
    # Returns masked logits (same shape)
    # soft: weight tools by probability of phases; hard: use argmax phase
    if hard:
        phase_ids = phase_probs.argmax(dim=-1)  # (B,)
        sample_mask = mask[phase_ids]          # (B, num_tools)
    else:
        # Soft mask: expected validity over phases
        # (B, P) x (P, T) -> (B, T)
        sample_mask = phase_probs @ mask

    # Where mask == 0, strongly penalize logits
    large_neg = -1e4
    masked_logits = tool_logits * sample_mask + large_neg * (1.0 - sample_mask)
    return masked_logits
