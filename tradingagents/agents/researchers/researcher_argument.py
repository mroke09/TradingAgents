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


def validate_argument_citations(
    argument: dict[str, Any] | None,
    specialist_findings: Any,
) -> dict[str, Any] | None:
    """Soft-validate debate evidence citations against the specialist blackboard."""

    if not isinstance(argument, dict):
        return argument
    valid_ids = _finding_ids(specialist_findings)
    evidence_items = argument.get("evidence")
    if not isinstance(evidence_items, list):
        return argument

    validated = dict(argument)
    next_evidence = []
    for item in evidence_items:
        if not isinstance(item, dict):
            next_evidence.append(item)
            continue
        next_item = dict(item)
        finding_id = next_item.get("finding_id")
        source = next_item.get("source")
        if finding_id:
            if finding_id in valid_ids:
                next_item["citation_status"] = "valid"
                next_item.pop("citation_note", None)
            else:
                next_item["citation_status"] = "invalid"
                next_item["citation_note"] = "Finding id was not present in specialist_findings."
                logger.warning("Invalid debate finding citation: %s", finding_id)
        elif source == "other":
            next_item["citation_status"] = "synthesized"
            next_item.setdefault(
                "citation_note",
                "Synthesized evidence without a single specialist finding id.",
            )
        else:
            next_item["citation_status"] = "uncited"
            next_item.setdefault(
                "citation_note",
                "Evidence did not cite a specialist finding id.",
            )
        next_evidence.append(next_item)
    validated["evidence"] = next_evidence
    return validated


def _finding_ids(specialist_findings: Any) -> set[str]:
    ids: set[str] = set()
    if not isinstance(specialist_findings, dict):
        return ids
    for findings in specialist_findings.values():
        if not isinstance(findings, list):
            continue
        for finding in findings:
            if isinstance(finding, dict) and finding.get("id"):
                ids.add(str(finding["id"]))
    return ids
