from __future__ import annotations

import pytest

from asr_service.config import Settings


def test_settings_parse_enabled_models_and_api_key() -> None:
    settings = Settings.from_env(
        {
            "ASR_ENABLED_MODELS": "0.6b,qwen3-asr-1.7b",
            "ASR_DOWNLOAD_SOURCE": "huggingface",
            "ASR_ENABLE_ALIGNER": "false",
            "ASR_API_KEY": " secret ",
        }
    )

    assert settings.accepted_model_aliases == ("qwen3-asr-0.6b", "qwen3-asr-1.7b")
    assert settings.download_source == "huggingface"
    assert settings.enable_aligner is False
    assert settings.api_key == "secret"


def test_settings_reject_invalid_download_source() -> None:
    with pytest.raises(ValueError, match="ASR_DOWNLOAD_SOURCE"):
        Settings.from_env({"ASR_DOWNLOAD_SOURCE": "invalid"})


def test_settings_resolve_only_enabled_model() -> None:
    settings = Settings.from_env({"ASR_ENABLED_MODELS": "0.6b"})

    assert settings.resolve_external_model("qwen3-asr-0.6b").key == "0.6b"

    with pytest.raises(ValueError, match="Enabled models"):
        settings.resolve_external_model("qwen3-asr-1.7b")
