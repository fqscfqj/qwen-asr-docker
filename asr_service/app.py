from __future__ import annotations

import contextlib
import os
import subprocess
import tempfile
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.datastructures import UploadFile

from .catalog import ALLOWED_RESPONSE_FORMATS, ALLOWED_TIMESTAMP_GRANULARITIES
from .config import Settings
from .formatters import build_segments, build_verbose_json, format_srt, format_vtt, words_from_time_stamps
from .language import normalize_language
from .model_manager import ModelManager


def _probe_duration_seconds(audio_path: Path) -> float | None:
    with contextlib.suppress(wave.Error):
        with wave.open(str(audio_path), "rb") as handle:
            frames = handle.getnframes()
            framerate = handle.getframerate()
            if framerate:
                return frames / float(framerate)

    with contextlib.suppress(Exception):
        import soundfile as sf

        info = sf.info(str(audio_path))
        if info.samplerate:
            return float(info.frames) / float(info.samplerate)

    with contextlib.suppress(Exception):
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        return float(result.stdout.strip())

    return None


def _collect_timestamp_granularities(form: Any) -> list[str]:
    raw_items = [*form.getlist("timestamp_granularities[]"), *form.getlist("timestamp_granularities")]
    values: list[str] = []
    for item in raw_items:
        normalized = str(item).strip().lower()
        if not normalized:
            continue
        if normalized not in ALLOWED_TIMESTAMP_GRANULARITIES:
            accepted = ", ".join(sorted(ALLOWED_TIMESTAMP_GRANULARITIES))
            raise HTTPException(status_code=400, detail=f"Unsupported timestamp_granularity {item!r}. Accepted values: {accepted}")
        if normalized not in values:
            values.append(normalized)
    return values


def _normalize_temperature(raw: Any) -> None:
    if raw is None or str(raw).strip() == "":
        return
    try:
        value = float(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="temperature must be empty or 0") from exc
    if value != 0:
        raise HTTPException(status_code=400, detail="temperature must be empty or 0")


def _require_auth(request: Request, settings: Settings) -> None:
    if not settings.api_key:
        return
    authorization = request.headers.get("Authorization", "")
    expected = f"Bearer {settings.api_key}"
    if authorization != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _build_openai_model_list(settings: Settings) -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {
                "id": spec.alias,
                "object": "model",
                "created": 0,
                "owned_by": "qwen",
            }
            for spec in settings.enabled_models
        ],
    }


def _app_settings(app: FastAPI) -> Settings:
    return app.state.settings


def _app_manager(app: FastAPI) -> ModelManager:
    return app.state.model_manager


def create_app(settings: Settings | None = None, model_manager: ModelManager | None = None) -> FastAPI:
    service_settings = settings or Settings.from_env()
    manager = model_manager or ModelManager(service_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await manager.start()
        try:
            yield
        finally:
            await manager.stop()

    app = FastAPI(title="Qwen3-ASR OpenAI-Compatible Service", lifespan=lifespan)
    app.state.settings = service_settings
    app.state.model_manager = manager

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, str]:
        if _app_manager(app).is_ready:
            return {"status": "ok"}
        raise HTTPException(status_code=503, detail="service not ready")

    @app.get("/v1/models")
    async def list_models(request: Request) -> JSONResponse:
        _require_auth(request, _app_settings(app))
        return JSONResponse(_build_openai_model_list(_app_settings(app)))

    @app.post("/v1/audio/transcriptions")
    async def create_transcription(request: Request):
        settings = _app_settings(app)
        manager = _app_manager(app)
        _require_auth(request, settings)

        form = await request.form()
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise HTTPException(status_code=400, detail="file is required")

        raw_model = str(form.get("model") or "").strip()
        if not raw_model:
            raise HTTPException(status_code=400, detail="model is required")
        try:
            spec = settings.resolve_external_model(raw_model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        response_format = str(form.get("response_format") or "json").strip().lower()
        if response_format not in ALLOWED_RESPONSE_FORMATS:
            accepted = ", ".join(sorted(ALLOWED_RESPONSE_FORMATS))
            raise HTTPException(status_code=400, detail=f"Unsupported response_format {response_format!r}. Accepted values: {accepted}")

        timestamp_granularities = _collect_timestamp_granularities(form)
        needs_timestamps = response_format in {"srt", "vtt"} or bool(timestamp_granularities)
        if needs_timestamps and not settings.enable_aligner:
            raise HTTPException(
                status_code=400,
                detail="timestamps require ASR_ENABLE_ALIGNER=true",
            )

        _normalize_temperature(form.get("temperature"))

        language = normalize_language(str(form.get("language") or "").strip() or None)
        prompt = str(form.get("prompt") or "").strip() or None

        suffix = Path(upload.filename or "audio.bin").suffix or ".bin"
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                payload = await upload.read()
                handle.write(payload)
                temp_path = Path(handle.name)

            result_list = await manager.transcribe(
                audio_path=str(temp_path),
                model_alias=spec.alias,
                language=language,
                prompt=prompt,
                return_time_stamps=needs_timestamps,
            )
            result = result_list[0]

            duration = _probe_duration_seconds(temp_path)
            words = words_from_time_stamps(getattr(result, "time_stamps", None)) if needs_timestamps else None
            segments = build_segments(words or []) if needs_timestamps else None

            if response_format == "text":
                return PlainTextResponse(str(result.text), media_type="text/plain; charset=utf-8")

            if response_format == "json":
                return JSONResponse({"text": str(result.text)})

            if response_format == "verbose_json":
                include_words = "word" in timestamp_granularities
                include_segments = "segment" in timestamp_granularities
                payload = build_verbose_json(
                    text=str(result.text),
                    language=str(result.language),
                    duration=duration,
                    words=words if include_words else None,
                    segments=segments if include_segments else None,
                )
                return JSONResponse(payload)

            if response_format == "srt":
                return PlainTextResponse(format_srt(segments or []), media_type="text/plain; charset=utf-8")

            return PlainTextResponse(format_vtt(segments or []), media_type="text/vtt; charset=utf-8")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            await upload.close()
            if temp_path is not None:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temp_path)

    return app


app = create_app()
