"""Deprecated compatibility wrapper for hierarchy utilities.

The canonical implementations now live in
``surgical_phase_tool.hierarchy.phase_tool_mask``. This module is kept
only so that any old imports like ``import hierarchy`` continue to
work without shipping duplicate logic.
"""

from surgical_phase_tool.hierarchy.phase_tool_mask import *                   
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
    return mask                           


def apply_phase_mask_to_logits(tool_logits: torch.Tensor, phase_probs: torch.Tensor, mask: torch.Tensor, hard: bool = False) -> torch.Tensor:
                                 
                                  
                                   
                                        
                                                                         
    if hard:
        phase_ids = phase_probs.argmax(dim=-1)        
        sample_mask = mask[phase_ids]                          
    else:
                                                  
                                   
        sample_mask = phase_probs @ mask

                                               
    large_neg = -1e4
    masked_logits = tool_logits * sample_mask + large_neg * (1.0 - sample_mask)
    return masked_logits
