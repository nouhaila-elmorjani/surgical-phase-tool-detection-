"""Compatibility wrapper for the multitask model.

Code should import PhaseToolNet from
``surgical_phase_tool.models.resnet_multitask``, but this module is kept
so that existing imports continue to work.
"""

from models.resnet_multitask import PhaseToolNet, TemporalAveragePool  # noqa: F401

