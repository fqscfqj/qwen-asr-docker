from __future__ import annotations

from collections.abc import Iterable, Sequence

from .catalog import SENTENCE_END_PUNCTUATION


def _contains_cjk(text: str) -> bool:
    return any(
        "\u4e00" <= ch <= "\u9fff"
        or "\u3400" <= ch <= "\u4dbf"
        or "\u3040" <= ch <= "\u30ff"
        or "\uac00" <= ch <= "\ud7af"
        for ch in text
    )


def _needs_space(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    if _contains_cjk(previous) or _contains_cjk(current):
        return False
    return previous[-1].isalnum() and current[0].isalnum()


def join_token_text(tokens: Sequence[str]) -> str:
    parts: list[str] = []
    for token in tokens:
        if not token:
            continue
        if parts and _needs_space(parts[-1], token):
            parts.append(" ")
        parts.append(token)
    return "".join(parts)


def words_from_time_stamps(time_stamps: Iterable[object] | None) -> list[dict[str, float | str]]:
    words: list[dict[str, float | str]] = []
    if not time_stamps:
        return words
    for item in time_stamps:
        word = str(getattr(item, "text", "")).strip()
        if not word:
            continue
        words.append(
            {
                "word": word,
                "start": round(float(getattr(item, "start_time", 0.0)), 3),
                "end": round(float(getattr(item, "end_time", 0.0)), 3),
            }
        )
    return words


def build_segments(words: Sequence[dict[str, float | str]]) -> list[dict[str, float | int | str]]:
    if not words:
        return []

    segments: list[dict[str, float | int | str]] = []
    current: list[dict[str, float | str]] = []

    def flush() -> None:
        if not current:
            return
        text = join_token_text([str(item["word"]) for item in current])
        segment = {
            "id": len(segments),
            "start": round(float(current[0]["start"]), 3),
            "end": round(float(current[-1]["end"]), 3),
            "text": text,
        }
        segments.append(segment)
        current.clear()

    for word in words:
        if current:
            previous = current[-1]
            gap = float(word["start"]) - float(previous["end"])
            next_text = join_token_text([*(str(item["word"]) for item in current), str(word["word"])])
            duration = float(word["end"]) - float(current[0]["start"])
            previous_word = str(previous["word"])
            if (
                previous_word.endswith(tuple(SENTENCE_END_PUNCTUATION))
                or gap > 0.8
                or duration > 12.0
                or len(next_text) > 120
            ):
                flush()
        current.append(word)

    flush()
    return segments


def _format_timestamp(seconds: float, separator: str) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def format_srt(segments: Sequence[dict[str, float | int | str]]) -> str:
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{_format_timestamp(float(segment['start']), ',')} --> {_format_timestamp(float(segment['end']), ',')}",
                    str(segment["text"]),
                ]
            )
        )
    return "\n\n".join(blocks)


def format_vtt(segments: Sequence[dict[str, float | int | str]]) -> str:
    lines = ["WEBVTT", ""]
    for segment in segments:
        lines.extend(
            [
                f"{_format_timestamp(float(segment['start']), '.')} --> {_format_timestamp(float(segment['end']), '.')}",
                str(segment["text"]),
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def build_verbose_json(
    *,
    text: str,
    language: str,
    duration: float | None,
    words: Sequence[dict[str, float | str]] | None,
    segments: Sequence[dict[str, float | int | str]] | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "text": text,
        "language": language,
        "duration": None if duration is None else round(duration, 3),
    }
    if words is not None:
        payload["words"] = list(words)
    if segments is not None:
        payload["segments"] = list(segments)
    return payload
