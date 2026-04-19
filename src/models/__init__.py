"""Model wrappers and compatibility shims for MiniMed Prime."""

from .trm_wrapper import (
    FallbackTinyRecursiveReasoningModel,
    SamsungTRMConfig,
    TRMModelBundle,
    load_real_trm,
    load_trm_model_bundle,
    stablemax_cross_entropy,
)

__all__ = [
    "FallbackTinyRecursiveReasoningModel",
    "SamsungTRMConfig",
    "TRMModelBundle",
    "load_real_trm",
    "load_trm_model_bundle",
    "stablemax_cross_entropy",
]
