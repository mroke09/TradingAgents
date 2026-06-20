"""Helpers for bull/bear researcher structured arguments."""

from __future__ import annotations

import logging
from typing import Any

from tradingagents.agents.schemas import (
    ResearcherArgument,
    render_researcher_argument,
)

logger = logging.getLogger(__name__)


def invoke_researcher_argument(
    *,
    structured_llm: Any | None,
    plain_llm: Any,
    prompt: str,
    agent_name: str,
) -> tuple[str, dict[str, Any] | None]:
    """Return rendered markdown plus optional structured argument data."""

    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            argument = (
                result
                if isinstance(result, ResearcherArgument)
                else ResearcherArgument.model_validate(result)
            )
            return render_researcher_argument(argument), argument.model_dump(mode="json")
        except Exception as exc:
            logger.warning(
                "%s: structured-output invocation failed (%s); retrying once as free text",
                agent_name,
                exc,
            )

    response = plain_llm.invoke(prompt)
    return response.content, None
