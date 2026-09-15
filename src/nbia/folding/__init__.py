from __future__ import annotations

from .base import FoldBackend, FoldJob, FoldPrediction
from .msa import (
    AbForgeGetOrBuildMSAProvider,
    BackendManagedMSAProvider,
    BackendMSASupport,
    MMseqs2MSAProvider,
    MSACache,
    MSACapability,
    MSAConfig,
    MSADeferredError,
    MSAError,
    MSAFormat,
    MSAMode,
    MSAPairing,
    MSAProvider,
    MSAProviderKind,
    MSAProvenance,
    MSARequest,
    MSAResolution,
    MSAStatus,
    MSAUnsupportedError,
    NoMSAProvider,
    PrecomputedMSAProvider,
    MsaInput,
    MsaPolicy,
    MsaRequirement,
)
from .boltz2_backend import Boltz2Backend
from .esmfold2_backend import ESMFold2Backend
from .opendde_backend import OpenDDEBackend


def get_fold_backend(name: str) -> FoldBackend:
    normalized = name.lower().replace("-", "_")
    if normalized in {"boltz", "boltz2"}:
        return Boltz2Backend()
    if normalized in {"esm", "esmfold2"}:
        return ESMFold2Backend()
    if normalized in {"opendde", "open_dde"}:
        return OpenDDEBackend()
    raise ValueError(f"Unknown fold backend: {name}")


__all__ = [
    "FoldBackend",
    "FoldJob",
    "FoldPrediction",
    "Boltz2Backend",
    "ESMFold2Backend",
    "OpenDDEBackend",
    "get_fold_backend",
    "AbForgeGetOrBuildMSAProvider",
    "BackendManagedMSAProvider",
    "BackendMSASupport",
    "MMseqs2MSAProvider",
    "MSACache",
    "MSACapability",
    "MSAConfig",
    "MSADeferredError",
    "MSAError",
    "MSAFormat",
    "MSAMode",
    "MSAPairing",
    "MSAProvider",
    "MSAProviderKind",
    "MSAProvenance",
    "MSARequest",
    "MSAResolution",
    "MSAStatus",
    "MSAUnsupportedError",
    "NoMSAProvider",
    "PrecomputedMSAProvider",
    "MsaInput",
    "MsaPolicy",
    "MsaRequirement",
]
