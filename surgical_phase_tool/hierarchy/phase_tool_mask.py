import torch

from ..dataset import PHASE_TO_ID, TOOL_COLUMNS


                                                            
                                

PHASE_TOOL_VALIDITY = {
    "Preparation": [
        0,               
        0,                   
        0,            
        0,           
        0,                        
        0,                
        0,                   
        1,              
        0,           
        0,               
        0,              
    ],
    "CalotTriangleDissection": [
        0,
        1,                   
        0,
        0,           
        0,
        0,
        0,
        1,              
        1,           
        0,
        0,
    ],
    "ClippingCutting": [
        0,
        1,                   
        0,
        0,
        0,
        0,
        0,
        0,
        1,           
        0,
        0,
    ],
    "GallbladderDissection": [
        0,
        0,
        0,
        0,
        1,                        
        0,
        0,
        1,              
        1,           
        1,               
        0,
    ],
    "CleaningCoagulation": [
        0,
        0,
        0,
        0,
        1,                        
        0,
        0,
        0,
        0,
        1,               
        0,
    ],
    "GallbladderPackaging": [
        0,
        0,
        0,
        0,
        0,
        1,                
        0,
        1,              
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
        1,              
        0,
        0,
        1,              
    ],
    "Undefined": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
    ],
}


def build_phase_tool_mask() -> torch.Tensor:
    """Build a (num_phases, num_tools) mask tensor.

    Entries are 1.0 for valid phase–tool combinations and 0.0 otherwise.
    """

    num_phases = len(PHASE_TO_ID)
    num_tools = len(TOOL_COLUMNS)
    mask = torch.zeros(num_phases, num_tools, dtype=torch.float32)
    for phase_name, phase_id in PHASE_TO_ID.items():
        valid_list = PHASE_TOOL_VALIDITY[phase_name]
        if len(valid_list) != num_tools:
            raise ValueError(
                f"Validity list for {phase_name} has wrong length {len(valid_list)}"
            )
        mask[phase_id] = torch.tensor(valid_list, dtype=torch.float32)
    return mask


def apply_phase_mask_to_logits(
    tool_logits: torch.Tensor,
    phase_probs: torch.Tensor,
    mask: torch.Tensor,
    hard: bool = False,
) -> torch.Tensor:
    """Apply phase→tool hierarchy to tool logits.

    Args:
        tool_logits: (B, num_tools)
        phase_probs: (B, num_phases)
        mask: (num_phases, num_tools) binary validity mask
        hard: if True, use argmax phase; if False, use soft expectation

    Returns:
        Masked logits of shape (B, num_tools) where invalid combinations
        have been strongly suppressed.
    """

    if hard:
        phase_ids = phase_probs.argmax(dim=-1)        
        sample_mask = mask[phase_ids]                          
    else:
                                                  
                                   
        sample_mask = phase_probs @ mask

    large_neg = -1e4
    masked_logits = tool_logits * sample_mask + large_neg * (1.0 - sample_mask)
    return masked_logits
