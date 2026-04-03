from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

from .catalog import ModelSpec
from .config import Settings


ArtifactRunner = Callable[[list[str]], None]


def has_model_artifacts(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    if not (model_dir / "config.json").exists():
        return False
    weight_patterns = ("*.safetensors", "*.bin", "*.pt", "*.pth")
    return any(next(model_dir.glob(pattern), None) is not None for pattern in weight_patterns)


def build_download_command(spec: ModelSpec, settings: Settings) -> list[str]:
    target_dir = settings.model_path(spec)
    if settings.download_source == "modelscope":
        return [
            "modelscope",
            "download",
            "--model",
            spec.repo_id,
            "--local_dir",
            str(target_dir),
        ]
    return [
        "huggingface-cli",
        "download",
        spec.repo_id,
        "--local-dir",
        str(target_dir),
    ]


def run_download_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def ensure_model_downloaded(spec: ModelSpec, settings: Settings, runner: ArtifactRunner = run_download_command) -> Path:
    target_dir = settings.model_path(spec)
    target_dir.mkdir(parents=True, exist_ok=True)
    if has_model_artifacts(target_dir):
        return target_dir
    runner(build_download_command(spec, settings))
    if not has_model_artifacts(target_dir):
        raise RuntimeError(f"Model download finished but required artifacts were not found in {target_dir}")
    return target_dir


def ensure_all_downloads(settings: Settings, runner: ArtifactRunner = run_download_command) -> None:
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    for spec in settings.enabled_models:
        ensure_model_downloaded(spec, settings, runner=runner)
    if settings.enable_aligner:
        ensure_model_downloaded(settings.aligner_spec, settings, runner=runner)
