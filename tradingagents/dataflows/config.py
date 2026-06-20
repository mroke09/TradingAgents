from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
from typing import Any, Callable, TypeVar

import tradingagents.default_config as default_config

# Use default config but allow it to be overridden
_config: dict | None = None
_config_lock = RLock()
_active_config: ContextVar[dict | None] = ContextVar(
    "tradingagents_active_config",
    default=None,
)
T = TypeVar("T")


def initialize_config():
    """Initialize the configuration with default values."""
    global _config
    with _config_lock:
        if _config is None:
            _config = deepcopy(default_config.DEFAULT_CONFIG)


def _merge_config(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    incoming = deepcopy(override)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def set_config(config: dict):
    """Update the process-default configuration with custom values.

    Dict-valued keys (e.g. ``data_vendors``) are merged one level deep so a
    partial update like ``{"data_vendors": {"core_stock_apis": "alpha_vantage"}}``
    keeps the other nested keys from the default; scalar keys are replaced.

    Analysis runs should prefer ``use_config`` so concurrent graphs do not share
    mutable process-global runtime configuration.
    """
    global _config
    initialize_config()
    with _config_lock:
        _config = _merge_config(_config, config)


def get_config() -> dict:
    """Get the active run configuration or the process default."""
    active = _active_config.get()
    if active is not None:
        return deepcopy(active)
    initialize_config()
    with _config_lock:
        return deepcopy(_config)


@contextmanager
def use_config(config: dict):
    """Temporarily bind config to the current execution context."""
    token = _active_config.set(deepcopy(config))
    try:
        yield
    finally:
        _active_config.reset(token)


def run_with_config(config: dict, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run ``func`` with ``config`` bound in this thread/context."""
    with use_config(config):
        return func(*args, **kwargs)


# Initialize with default config
initialize_config()
