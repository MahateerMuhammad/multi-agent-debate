from enum import Enum


class StopReason(str, Enum):
    """Canonical reasons for graph termination."""

    direct_execution = "direct_execution"
    confidence_threshold = "confidence_threshold"
    quality_converged = "quality_converged"
    max_rounds = "max_rounds"
    fatal_system_error = "fatal_system_error"
