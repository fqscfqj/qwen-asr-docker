from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    alias: str
    repo_id: str
    local_dir_name: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "0.6b": ModelSpec(
        key="0.6b",
        alias="qwen3-asr-0.6b",
        repo_id="Qwen/Qwen3-ASR-0.6B",
        local_dir_name="Qwen3-ASR-0.6B",
    ),
    "1.7b": ModelSpec(
        key="1.7b",
        alias="qwen3-asr-1.7b",
        repo_id="Qwen/Qwen3-ASR-1.7B",
        local_dir_name="Qwen3-ASR-1.7B",
    ),
}

ALIGNER_SPEC = ModelSpec(
    key="aligner",
    alias="qwen3-forced-aligner-0.6b",
    repo_id="Qwen/Qwen3-ForcedAligner-0.6B",
    local_dir_name="Qwen3-ForcedAligner-0.6B",
)

ALLOWED_DOWNLOAD_SOURCES = frozenset({"modelscope", "huggingface"})
ALLOWED_RESPONSE_FORMATS = frozenset({"json", "text", "verbose_json", "srt", "vtt"})
ALLOWED_TIMESTAMP_GRANULARITIES = frozenset({"word", "segment"})
SENTENCE_END_PUNCTUATION = frozenset({"。", "！", "？", ".", "!", "?", "；", ";"})
