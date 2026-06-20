from __future__ import annotations

from typing import Any


def configured_skill_prompt(config: dict[str, Any] | None, agent_id: str) -> str | None:
    if not config:
        return None
    skill = (config.get("skills") or {}).get(agent_id) or {}
    prompt = skill.get("prompt")
    return prompt if isinstance(prompt, str) and prompt.strip() else None


def render_configured_skill_prompt(
    *,
    config: dict[str, Any] | None,
    agent_id: str,
    context: dict[str, Any],
) -> str | None:
    prompt = configured_skill_prompt(config, agent_id)
    if not prompt:
        return None
    return prompt.format(**context)
