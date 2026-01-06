"""Compatibility wrapper for the multitask model.

Preferred import path is
``surgical_phase_tool.models.resnet_multitask.PhaseToolNet`` but this
module is kept so that existing imports continue to work.
"""

from .models.resnet_multitask import PhaseToolNet, TemporalAveragePool              

