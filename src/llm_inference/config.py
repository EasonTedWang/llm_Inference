"""Configuration models for the inference engine skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BackendKind(str, Enum):
    """Selectable execution backend."""

    CUDA = "cuda"
    CPU = "cpu"
    MOCK = "mock"

    @classmethod
    def parse(cls, value: str | "BackendKind") -> "BackendKind":
        if isinstance(value, cls):
            return value
        normalized = value.lower().strip()
        for item in cls:
            if item.value == normalized:
                return item
        allowed = ", ".join(item.value for item in cls)
        raise ValueError(f"unknown backend {value!r}; expected one of: {allowed}")


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "mock-model"
    model_path: str | None = None
    tokenizer_path: str | None = None


@dataclass(frozen=True)
class SchedulerConfig:
    max_batch_size: int = 1
    enable_continuous_batching: bool = False

    def validate(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")


@dataclass(frozen=True)
class MemoryConfig:
    kv_block_size: int = 16
    max_kv_blocks: int = 1024
    enable_prefix_cache: bool = False

    def validate(self) -> None:
        if self.kv_block_size < 1:
            raise ValueError("kv_block_size must be >= 1")
        if self.max_kv_blocks < 1:
            raise ValueError("max_kv_blocks must be >= 1")


@dataclass(frozen=True)
class ObservabilityConfig:
    log_level: str = "INFO"
    enable_audit_events: bool = True


@dataclass(frozen=True)
class EngineConfig:
    backend: BackendKind = BackendKind.MOCK
    model: ModelConfig = field(default_factory=ModelConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    @classmethod
    def for_backend(cls, backend: str | BackendKind) -> "EngineConfig":
        return cls(backend=BackendKind.parse(backend))

    def validate(self) -> None:
        BackendKind.parse(self.backend)
        self.scheduler.validate()
        self.memory.validate()

