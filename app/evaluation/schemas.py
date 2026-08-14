from enum import Enum


class EvaluationMode(str, Enum):
    """Canonical evaluation modes."""

    synthetic = "synthetic"
    real_model = "real_model"


class UsageSource(str, Enum):
    """Provenance of token usage metrics."""

    provider_reported = "provider_reported"
    simulated = "simulated"
    unavailable = "unavailable"


class CostSource(str, Enum):
    """Provenance of cost calculation."""

    provider_reported = "provider_reported"
    simulated = "simulated"
    local_pricing_estimate = "local_pricing_estimate"
    unavailable = "unavailable"


class ConfidenceStatus(str, Enum):
    """Status of self-reported confidence."""

    reported = "reported"
    missing = "missing"
    invalid = "invalid"


class CorrectnessStatus(str, Enum):
    """Evaluation status for generative answer correctness."""

    evaluated = "evaluated"
    not_evaluable = "not_evaluable"
