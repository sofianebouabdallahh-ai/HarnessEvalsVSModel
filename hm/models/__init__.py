from hm.models.base import ModelBackend, ModelResponse, ToolCall
from hm.types import ModelSpec


def make_backend(spec: ModelSpec) -> ModelBackend:
    if spec.provider == "mock":
        from hm.models.mock import MockModel

        return MockModel(spec.name)
    if spec.provider == "anthropic":
        from hm.models.anthropic import AnthropicModel

        return AnthropicModel(spec.name)
    raise ValueError(f"unknown provider {spec.provider!r}")


__all__ = ["ModelBackend", "ModelResponse", "ToolCall", "make_backend"]
