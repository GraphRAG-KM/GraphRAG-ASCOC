import os
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Union


class PipelineStats:
    """Lightweight tracker for pipeline metrics."""

    def __init__(self) -> None:
        self._timers: Dict[str, float] = {}
        self.timings: Dict[str, float] = defaultdict(float)
        self.file_sizes: Dict[str, int] = {}
        self.token_usage: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "total": 0}
        )

    # ------------------------------------------------------------------
    # Size helpers
    def record_file_size(
        self, label: str, path: Optional[Union[str, Path]] = None, size_bytes: Optional[int] = None
    ) -> None:
        """Record the size of a file or arbitrary payload."""
        if size_bytes is None:
            if path is None:
                return
            try:
                size_bytes = os.path.getsize(path)
            except OSError:
                return
        self.file_sizes[label] = int(size_bytes)

    # ------------------------------------------------------------------
    # Timing helpers
    def start_timer(self, label: str) -> None:
        self._timers[label] = time.perf_counter()

    def stop_timer(self, label: str) -> Optional[float]:
        start = self._timers.pop(label, None)
        if start is None:
            return None
        duration = time.perf_counter() - start
        self.timings[label] += duration
        return duration

    def add_duration(self, label: str, seconds: float) -> None:
        self.timings[label] += max(0.0, seconds)

    # ------------------------------------------------------------------
    # Token helpers
    def record_token_usage(
        self, model: str, usage: Optional[Any], label: Optional[str] = None
    ) -> None:
        """Record token usage for an OpenAI response."""
        if usage is None:
            return

        usage_dict = self._usage_to_dict(usage)
        if not usage_dict:
            return

        prompt = self._first_present(usage_dict, ("prompt_tokens", "input_tokens"))
        completion = self._first_present(
            usage_dict, ("completion_tokens", "output_tokens")
        )
        total = usage_dict.get("total_tokens")

        prompt = int(prompt) if prompt is not None else 0
        completion = int(completion) if completion is not None else 0

        if total is None:
            total = prompt + completion
        total = int(total)

        entry = self.token_usage[model]
        entry["prompt"] += prompt
        entry["completion"] += completion
        entry["total"] += total

        if label:
            entry_key = f"{model}:{label}"
            extra = self.token_usage[entry_key]
            extra["prompt"] += prompt
            extra["completion"] += completion
            extra["total"] += total

        overall = self.token_usage["_total"]
        overall["prompt"] += prompt
        overall["completion"] += completion
        overall["total"] += total

    @staticmethod
    def _usage_to_dict(usage: Any) -> Dict[str, Any]:
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return usage
        for attr in ("model_dump", "dict", "to_dict"):
            if hasattr(usage, attr):
                try:
                    data = getattr(usage, attr)()
                    if isinstance(data, dict):
                        return data
                except Exception:
                    continue
        if hasattr(usage, "__dict__"):
            return dict(vars(usage))
        return {}

    @staticmethod
    def _first_present(data: Dict[str, Any], keys: tuple[str, ...]) -> Optional[int]:
        for key in keys:
            value = data.get(key)
            if value is not None:
                return value
        return None

    # ------------------------------------------------------------------
    # Reporting
    def formatted_summary(self) -> list[str]:
        lines: list[str] = []

        if self.file_sizes:
            lines.append("[cyan]Document Sizes[/]")
            for label, size in self.file_sizes.items():
                lines.append(f"  • {label}: {self._format_bytes(size)}")

        if self.timings:
            lines.append("[cyan]Timings[/]")
            for label, seconds in self.timings.items():
                lines.append(f"  • {label}: {seconds:.2f}s")

        # Filter special aggregate key
        token_keys = [k for k in self.token_usage.keys() if k != "_total"]
        overall = self.token_usage.get("_total")
        if token_keys or overall:
            lines.append("[cyan]Token Usage[/]")
            for key in sorted(token_keys):
                if key == "_total":
                    continue
                usage = self.token_usage[key]
                lines.append(
                    f"  • {key}: prompt {usage['prompt']}, completion {usage['completion']}, total {usage['total']}"
                )
            if overall:
                lines.append(
                    f"  • overall: prompt {overall['prompt']}, completion {overall['completion']}, total {overall['total']}"
                )

        return lines

    @staticmethod
    def _format_bytes(num: int) -> str:
        step = 1024.0
        units = ["B", "KB", "MB", "GB", "TB"]
        value = float(num)
        for unit in units:
            if value < step or unit == units[-1]:
                return f"{value:.2f} {unit}"
            value /= step
        return f"{value:.2f} TB"


pipeline_stats = PipelineStats()


@contextmanager
def capture_openai_usage(label: str):
    """Temporarily instrument OpenAI client calls to capture token usage."""
    try:
        from openai.resources.chat.completions import Completions as SyncChatCompletions
        from openai.resources.chat.completions import (
            AsyncCompletions as AsyncChatCompletions,
        )
        from openai.resources.embeddings import Embeddings as SyncEmbeddings
        from openai.resources.embeddings import AsyncEmbeddings as AsyncEmbeddings
        from openai.resources.responses import Responses as SyncResponses
        from openai.resources.responses import AsyncResponses as AsyncResponses
    except Exception:
        # OpenAI package not available; do nothing
        yield
        return

    original_sync_chat = getattr(SyncChatCompletions, "create", None)
    original_async_chat = getattr(AsyncChatCompletions, "create", None)
    original_sync_embed = getattr(SyncEmbeddings, "create", None)
    original_async_embed = getattr(AsyncEmbeddings, "create", None)
    original_sync_response = getattr(SyncResponses, "create", None)
    original_async_response = getattr(AsyncResponses, "create", None)

    def _extract_model(args, kwargs, response) -> str:
        return (
            kwargs.get("model")
            or getattr(response, "model", None)
            or getattr(response, "meta", {}).get("model")
            or "unknown"
        )

    def _record(response, model: str, kind: str):
        pipeline_stats.record_token_usage(
            str(model),
            getattr(response, "usage", None),
            f"{label}:{kind}",
        )

    def sync_chat_wrapper(self, *args, **kwargs):
        response = original_sync_chat(self, *args, **kwargs)
        _record(response, _extract_model(args, kwargs, response), "chat")
        return response

    async def async_chat_wrapper(self, *args, **kwargs):
        response = await original_async_chat(self, *args, **kwargs)
        _record(response, _extract_model(args, kwargs, response), "chat")
        return response

    def sync_embed_wrapper(self, *args, **kwargs):
        response = original_sync_embed(self, *args, **kwargs)
        _record(response, _extract_model(args, kwargs, response), "embedding")
        return response

    async def async_embed_wrapper(self, *args, **kwargs):
        response = await original_async_embed(self, *args, **kwargs)
        _record(response, _extract_model(args, kwargs, response), "embedding")
        return response

    def sync_response_wrapper(self, *args, **kwargs):
        response = original_sync_response(self, *args, **kwargs)
        _record(response, _extract_model(args, kwargs, response), "response")
        return response

    async def async_response_wrapper(self, *args, **kwargs):
        response = await original_async_response(self, *args, **kwargs)
        _record(response, _extract_model(args, kwargs, response), "response")
        return response

    # Apply patches
    if original_sync_chat is not None:
        SyncChatCompletions.create = sync_chat_wrapper  # type: ignore[assignment]
    if original_async_chat is not None:
        AsyncChatCompletions.create = async_chat_wrapper  # type: ignore[assignment]
    if original_sync_embed is not None:
        SyncEmbeddings.create = sync_embed_wrapper  # type: ignore[assignment]
    if original_async_embed is not None:
        AsyncEmbeddings.create = async_embed_wrapper  # type: ignore[assignment]
    if original_sync_response is not None:
        SyncResponses.create = sync_response_wrapper  # type: ignore[assignment]
    if original_async_response is not None:
        AsyncResponses.create = async_response_wrapper  # type: ignore[assignment]

    try:
        yield
    finally:
        if original_sync_chat is not None:
            SyncChatCompletions.create = original_sync_chat  # type: ignore[assignment]
        if original_async_chat is not None:
            AsyncChatCompletions.create = original_async_chat  # type: ignore[assignment]
        if original_sync_embed is not None:
            SyncEmbeddings.create = original_sync_embed  # type: ignore[assignment]
        if original_async_embed is not None:
            AsyncEmbeddings.create = original_async_embed  # type: ignore[assignment]
        if original_sync_response is not None:
            SyncResponses.create = original_sync_response  # type: ignore[assignment]
        if original_async_response is not None:
            AsyncResponses.create = original_async_response  # type: ignore[assignment]
