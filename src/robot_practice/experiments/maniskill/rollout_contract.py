"""Runtime rollout-backend contracts for ManiSkill experiment data."""

from __future__ import annotations

from dataclasses import dataclass

OFFICIAL_DATA_PROVENANCE = "official"
M1W_DATA_PROVENANCE = "m1w"
PHYSX_CPU = "physx_cpu"
PHYSX_CUDA = "physx_cuda"

_PROVENANCE_ALIASES = {
    "official": OFFICIAL_DATA_PROVENANCE,
    "official_rgb_18d_4d": OFFICIAL_DATA_PROVENANCE,
    "motionplanning": OFFICIAL_DATA_PROVENANCE,
    "paper_aligned_4d": OFFICIAL_DATA_PROVENANCE,
    "m1w": M1W_DATA_PROVENANCE,
    "m1w_live_rgb": M1W_DATA_PROVENANCE,
    "m1w_projected_rgb_4d": M1W_DATA_PROVENANCE,
    "native4d": M1W_DATA_PROVENANCE,
    "m1_w_native_4d": M1W_DATA_PROVENANCE,
}


@dataclass(frozen=True)
class RolloutContract:
    """The simulator backend required by a dataset provenance class."""

    data_provenance: str
    required_rollout_backend: str

    def verify(self, actual_backend: str) -> dict[str, str]:
        actual = normalize_rollout_backend(actual_backend)
        if actual != self.required_rollout_backend:
            raise ValueError(
                "rollout backend violates the dataset contract: "
                f"data_provenance={self.data_provenance!r} requires "
                f"{self.required_rollout_backend!r}, got {actual!r}"
            )
        return {
            "data_provenance": self.data_provenance,
            "required_rollout_backend": self.required_rollout_backend,
            "actual_rollout_backend": actual,
        }


def normalize_data_provenance(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    try:
        return _PROVENANCE_ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_PROVENANCE_ALIASES))
        raise ValueError(
            f"unknown data provenance {value!r}; supported values/aliases: {supported}"
        ) from exc


def normalize_rollout_backend(value: str) -> str:
    key = str(value).strip().lower()
    if key in {"cpu", PHYSX_CPU}:
        return PHYSX_CPU
    if key in {"gpu", PHYSX_CUDA} or key.startswith(f"{PHYSX_CUDA}:"):
        return PHYSX_CUDA
    raise ValueError(
        f"unknown rollout backend {value!r}; expected {PHYSX_CPU!r} or {PHYSX_CUDA!r}"
    )


def rollout_contract(data_provenance: str) -> RolloutContract:
    provenance = normalize_data_provenance(data_provenance)
    required = PHYSX_CPU if provenance == OFFICIAL_DATA_PROVENANCE else PHYSX_CUDA
    return RolloutContract(provenance, required)


def require_rollout_backend(data_provenance: str, actual_backend: str) -> dict[str, str]:
    """Validate a rollout backend and return portable manifest metadata."""

    return rollout_contract(data_provenance).verify(actual_backend)


__all__ = [
    "M1W_DATA_PROVENANCE",
    "OFFICIAL_DATA_PROVENANCE",
    "PHYSX_CPU",
    "PHYSX_CUDA",
    "RolloutContract",
    "normalize_data_provenance",
    "normalize_rollout_backend",
    "require_rollout_backend",
    "rollout_contract",
]
