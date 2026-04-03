from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import time
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .downloader import ensure_all_downloads

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ActiveModel:
    spec_alias: str
    wrapper: Any


class ModelManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._state_lock = asyncio.Lock()
        self._state_changed = asyncio.Condition(self._state_lock)
        self._inference_lock = asyncio.Lock()
        self._active: ActiveModel | None = None
        self._in_flight = 0
        self._last_used_at = 0.0
        self._ready = False
        self._reaper_task: asyncio.Task[None] | None = None

    @property
    def is_ready(self) -> bool:
        return self._ready

    async def start(self) -> None:
        if self._ready:
            return
        await asyncio.to_thread(ensure_all_downloads, self.settings)
        self._ready = True
        if self._reaper_task is None:
            self._reaper_task = asyncio.create_task(self._idle_reaper())

    async def stop(self) -> None:
        task = self._reaper_task
        self._reaper_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        async with self._state_changed:
            while self._in_flight > 0:
                await self._state_changed.wait()
            await asyncio.to_thread(self._unload_sync)
        self._ready = False

    async def transcribe(
        self,
        *,
        audio_path: str,
        model_alias: str,
        language: str | None,
        prompt: str | None,
        return_time_stamps: bool,
    ) -> Any:
        wrapper = await self._acquire(model_alias, return_time_stamps)
        try:
            async with self._inference_lock:
                result = await asyncio.to_thread(
                    wrapper.transcribe,
                    audio=audio_path,
                    context=prompt or "",
                    language=language,
                    return_time_stamps=return_time_stamps,
                )
                return result
        finally:
            await self._release()

    async def _acquire(self, model_alias: str, require_aligner: bool) -> Any:
        async with self._state_changed:
            while self._active is not None and self._active.spec_alias != model_alias and self._in_flight > 0:
                await self._state_changed.wait()

            if self._active is not None and self._active.spec_alias != model_alias:
                await asyncio.to_thread(self._unload_sync)

            if self._active is None:
                self._active = ActiveModel(
                    spec_alias=model_alias,
                    wrapper=await asyncio.to_thread(self._load_model_sync, model_alias),
                )

            if require_aligner and getattr(self._active.wrapper, "forced_aligner", None) is None:
                while self._in_flight > 0:
                    await self._state_changed.wait()
                await asyncio.to_thread(self._load_aligner_sync, self._active.wrapper)

            self._in_flight += 1
            self._last_used_at = time.monotonic()
            return self._active.wrapper

    async def _release(self) -> None:
        async with self._state_changed:
            self._in_flight = max(0, self._in_flight - 1)
            self._last_used_at = time.monotonic()
            self._state_changed.notify_all()

    async def _idle_reaper(self) -> None:
        sleep_seconds = 5 if self.settings.idle_unload_seconds == 0 else min(max(self.settings.idle_unload_seconds / 2, 1), 30)
        try:
            while True:
                await asyncio.sleep(sleep_seconds)
                async with self._state_changed:
                    if self._active is None or self._in_flight > 0:
                        continue
                    idle_for = time.monotonic() - self._last_used_at
                    if idle_for < self.settings.idle_unload_seconds:
                        continue
                    await asyncio.to_thread(self._unload_sync)
        except asyncio.CancelledError:
            raise

    def _load_model_sync(self, model_alias: str) -> Any:
        spec = self.settings.resolve_external_model(model_alias)
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise RuntimeError("qwen-asr is not installed. Use the Docker image or install qwen-asr[vllm].") from exc

        model = Qwen3ASRModel.LLM(
            model=str(self.settings.model_path(spec)),
            gpu_memory_utilization=self.settings.gpu_memory_utilization,
            max_inference_batch_size=self.settings.max_inference_batch_size,
            max_new_tokens=self.settings.max_new_tokens,
        )
        return model

    def _load_aligner_sync(self, wrapper: Any) -> None:
        if getattr(wrapper, "forced_aligner", None) is not None:
            return
        try:
            import torch
            from qwen_asr import Qwen3ForcedAligner
        except ImportError as exc:
            raise RuntimeError("Forced aligner dependencies are not installed.") from exc

        dtype = getattr(torch, self.settings.aligner_dtype, None)
        if dtype is None:
            raise RuntimeError(f"Unsupported torch dtype for aligner: {self.settings.aligner_dtype}")

        wrapper.forced_aligner = Qwen3ForcedAligner.from_pretrained(
            str(self.settings.aligner_path),
            dtype=dtype,
            device_map=self.settings.aligner_device_map,
        )

    def _unload_sync(self) -> None:
        if self._active is None:
            return

        wrapper = self._active.wrapper
        self._active = None

        if hasattr(wrapper, "forced_aligner"):
            wrapper.forced_aligner = None

        if hasattr(wrapper, "model"):
            wrapper.model = None

        del wrapper
        gc.collect()

        try:
            import torch

            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
        except Exception:
            logger.exception("torch cache cleanup failed")

        try:
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

            cleanup_dist_env_and_memory()
        except Exception:
            logger.debug("vLLM distributed cleanup skipped", exc_info=True)
