"""Deprecated compatibility wrapper for metrics utilities.

The canonical implementations now live in
``surgical_phase_tool.metrics``. This module is kept only so that any
old imports like ``import metrics`` continue to work without shipping
duplicate logic.
"""

from surgical_phase_tool.metrics import *                   
