from enum import Enum


class EvaluationMode(str, Enum):
    """Canonical evaluation modes."""
    synthetic = "synthetic"
    real_model = "real_model"


class UsageSource(str, Enum):
    """Provenance of token usage metrics."""
    provider_reported = "provider_reported"
    simulated = "simulated"
    estimated = "estimated"
    unavailable = "unavailable"


class CorrectnessStatus(str, Enum):
    """Evaluation status for generative answer correctness."""
    evaluated = "evaluated"
    not_evaluable = "not_evaluable"
