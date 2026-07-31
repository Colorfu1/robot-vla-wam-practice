"""Offline LIBERO asset validation and runtime configuration."""

from __future__ import annotations

from pathlib import Path

REQUIRED_ASSET_DIRECTORIES = (
    "articulated_objects",
    "scenes",
    "stable_hope_objects",
    "stable_scanned_objects",
    "textures",
    "turbosquid_objects",
)
REQUIRED_ASSET_FILES = ("scenes/libero_tabletop_base_style.xml",)


def validate_libero_assets_dir(path: str | Path) -> Path:
    assets_dir = Path(path).expanduser().resolve()
    if not assets_dir.is_dir():
        raise FileNotFoundError(
            f"LIBERO assets directory does not exist: {assets_dir}. "
            "Provide an offline copy with --libero-assets-dir."
        )
    missing_directories = [
        name for name in REQUIRED_ASSET_DIRECTORIES if not (assets_dir / name).is_dir()
    ]
    missing_files = [name for name in REQUIRED_ASSET_FILES if not (assets_dir / name).is_file()]
    if missing_directories or missing_files:
        details = [
            *(f"directory:{name}" for name in missing_directories),
            *(f"file:{name}" for name in missing_files),
        ]
        raise FileNotFoundError(
            f"Incomplete LIBERO assets directory {assets_dir}; missing {', '.join(details)}"
        )
    return assets_dir


def configure_libero_assets_dir(path: str | Path) -> Path:
    """Force hf-libero to use a validated offline asset tree without Hub access."""

    assets_dir = validate_libero_assets_dir(path)
    import libero.libero as libero_root

    libero_root._assets_path_cache = str(assets_dir)
    return assets_dir


__all__ = [
    "REQUIRED_ASSET_DIRECTORIES",
    "REQUIRED_ASSET_FILES",
    "configure_libero_assets_dir",
    "validate_libero_assets_dir",
]
