from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .catalog import ALIGNER_SPEC, ALLOWED_DOWNLOAD_SOURCES, MODEL_SPECS, ModelSpec


def _parse_bool(name: str, value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got: {value!r}")


def _parse_int(name: str, value: str | None, default: int, minimum: int | None = None) -> int:
    raw = default if value is None or value.strip() == "" else int(value)
    if minimum is not None and raw < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got: {raw}")
    return raw


def _parse_float(
    name: str,
    value: str | None,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = default if value is None or value.strip() == "" else float(value)
    if minimum is not None and raw < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got: {raw}")
    if maximum is not None and raw > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got: {raw}")
    return raw


def _parse_enabled_models(raw: str | None) -> tuple[ModelSpec, ...]:
    value = raw or "0.6b,1.7b"
    enabled: list[ModelSpec] = []
    seen: set[str] = set()
    for token in value.split(","):
        normalized = token.strip().lower()
        if not normalized:
            continue
        spec = MODEL_SPECS.get(normalized)
        if spec is None:
            spec = next((item for item in MODEL_SPECS.values() if item.alias == normalized), None)
        if spec is None:
            accepted = ", ".join(sorted([*MODEL_SPECS.keys(), *(item.alias for item in MODEL_SPECS.values())]))
            raise ValueError(f"Unsupported ASR model {token!r}. Accepted values: {accepted}")
        if spec.key not in seen:
            enabled.append(spec)
            seen.add(spec.key)
    if not enabled:
        raise ValueError("ASR_ENABLED_MODELS must enable at least one model")
    return tuple(enabled)


@dataclass(frozen=True, slots=True)
class Settings:
    enabled_models: tuple[ModelSpec, ...]
    download_source: str
    enable_aligner: bool
    idle_unload_seconds: int
    gpu_memory_utilization: float
    api_key: str | None
    models_dir: Path
    port: int
    host: str
    max_inference_batch_size: int
    max_new_tokens: int
    aligner_device_map: str
    aligner_dtype: str

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ

        enabled_models = _parse_enabled_models(env.get("ASR_ENABLED_MODELS"))
        download_source = (env.get("ASR_DOWNLOAD_SOURCE") or "modelscope").strip().lower()
        if download_source not in ALLOWED_DOWNLOAD_SOURCES:
            accepted = ", ".join(sorted(ALLOWED_DOWNLOAD_SOURCES))
            raise ValueError(f"ASR_DOWNLOAD_SOURCE must be one of: {accepted}")

        api_key = (env.get("ASR_API_KEY") or "").strip() or None

        return cls(
            enabled_models=enabled_models,
            download_source=download_source,
            enable_aligner=_parse_bool("ASR_ENABLE_ALIGNER", env.get("ASR_ENABLE_ALIGNER"), True),
            idle_unload_seconds=_parse_int(
                "ASR_IDLE_UNLOAD_SECONDS",
                env.get("ASR_IDLE_UNLOAD_SECONDS"),
                900,
                minimum=0,
            ),
            gpu_memory_utilization=_parse_float(
                "ASR_GPU_MEMORY_UTILIZATION",
                env.get("ASR_GPU_MEMORY_UTILIZATION"),
                0.8,
                minimum=0.05,
                maximum=1.0,
            ),
            api_key=api_key,
            models_dir=Path(env.get("ASR_MODELS_DIR") or "/models"),
            port=_parse_int("ASR_PORT", env.get("ASR_PORT"), 8000, minimum=1),
            host=(env.get("ASR_HOST") or "0.0.0.0").strip() or "0.0.0.0",
            max_inference_batch_size=_parse_int(
                "ASR_MAX_INFERENCE_BATCH_SIZE",
                env.get("ASR_MAX_INFERENCE_BATCH_SIZE"),
                32,
                minimum=1,
            ),
            max_new_tokens=_parse_int("ASR_MAX_NEW_TOKENS", env.get("ASR_MAX_NEW_TOKENS"), 4096, minimum=1),
            aligner_device_map=(env.get("ASR_ALIGNER_DEVICE_MAP") or "cuda:0").strip() or "cuda:0",
            aligner_dtype=(env.get("ASR_ALIGNER_DTYPE") or "bfloat16").strip() or "bfloat16",
        )

    @property
    def aligner_spec(self) -> ModelSpec:
        return ALIGNER_SPEC

    @property
    def accepted_model_aliases(self) -> tuple[str, ...]:
        return tuple(spec.alias for spec in self.enabled_models)

    def resolve_external_model(self, value: str) -> ModelSpec:
        normalized = value.strip().lower()
        for spec in self.enabled_models:
            if spec.alias == normalized:
                return spec
        accepted = ", ".join(self.accepted_model_aliases)
        raise ValueError(f"Unsupported model {value!r}. Enabled models: {accepted}")

    def model_path(self, spec: ModelSpec) -> Path:
        return self.models_dir / spec.local_dir_name

    @property
    def aligner_path(self) -> Path:
        return self.model_path(self.aligner_spec)
