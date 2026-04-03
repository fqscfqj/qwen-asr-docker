from __future__ import annotations

from asr_service.formatters import build_segments, format_srt, format_vtt, join_token_text


def test_join_token_text_handles_english_spacing() -> None:
    assert join_token_text(["hello", "world", "!"]) == "hello world!"
    assert join_token_text(["你", "好"]) == "你好"


def test_build_segments_splits_on_gap_and_punctuation() -> None:
    words = [
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "world.", "start": 0.5, "end": 1.0},
        {"word": "again", "start": 2.2, "end": 2.7},
    ]

    segments = build_segments(words)

    assert [segment["text"] for segment in segments] == ["hello world.", "again"]


def test_srt_and_vtt_render_expected_timestamps() -> None:
    segments = [{"id": 0, "start": 1.234, "end": 3.5, "text": "sample"}]

    assert "00:00:01,234 --> 00:00:03,500" in format_srt(segments)
    assert "WEBVTT" in format_vtt(segments)
    assert "00:00:01.234 --> 00:00:03.500" in format_vtt(segments)
