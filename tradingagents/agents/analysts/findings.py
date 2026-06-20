"""Structured finding extraction for specialist analyst reports."""

from __future__ import annotations

import logging
from typing import Any, Literal

from tradingagents.agents.schemas import (
    SpecialistFinding,
    SpecialistFindingsReport,
)
from tradingagents.agents.utils.structured import bind_structured

logger = logging.getLogger(__name__)

SpecialistKey = Literal["market", "fundamentals", "news", "sentiment"]


def bind_findings_extractor(llm: Any, agent_name: str) -> Any | None:
    return bind_structured(llm, SpecialistFindingsReport, f"{agent_name} Findings")


def extract_specialist_findings(
    structured_llm: Any | None,
    *,
    agent_key: SpecialistKey,
    report_text: str,
    trade_date: str,
    instrument_context: str,
) -> list[dict[str, Any]]:
    """Extract structured findings from a completed specialist markdown report."""

    if structured_llm is None or not report_text.strip():
        return []

    prompt = f"""Extract structured specialist findings from the analyst report below.

Rules:
- Produce only findings grounded in the report text. Do not invent metrics, values, dates, or sources.
- Each finding must have an id unique within this analyst, prefixed with "{agent_key}-".
- The `agent` field must be exactly "{agent_key}".
- Prefer high-signal claims over generic summaries.
- Use direction to describe investment implication: bullish / bearish / neutral / mixed.
- Use confidence to reflect data quality and consistency, not rhetorical strength.

Instrument context:
{instrument_context}

Analysis date: {trade_date}

Specialist report:
{report_text}
"""
    try:
        result = structured_llm.invoke(prompt)
        findings_report = (
            result
            if isinstance(result, SpecialistFindingsReport)
            else SpecialistFindingsReport.model_validate(result)
        )
    except Exception as exc:
        logger.warning(
            "%s findings extraction failed (%s); continuing without structured findings",
            agent_key,
            exc,
        )
        return []

    return [
        _normalize_finding(finding, agent_key, trade_date, index)
        for index, finding in enumerate(findings_report.findings, start=1)
    ]


def _normalize_finding(
    finding: SpecialistFinding,
    agent_key: SpecialistKey,
    trade_date: str,
    index: int,
) -> dict[str, Any]:
    data = finding.model_dump(mode="json")
    data["agent"] = agent_key
    if not data.get("id"):
        data["id"] = f"{agent_key}-{index}"
    elif not str(data["id"]).startswith(f"{agent_key}-"):
        data["id"] = f"{agent_key}-{data['id']}"
    data["as_of_date"] = data.get("as_of_date") or trade_date
    return data


def update_specialist_findings(
    state: dict[str, Any],
    agent_key: SpecialistKey,
    findings: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return a new specialist_findings blackboard with one agent slot updated."""

    current = dict(state.get("specialist_findings") or {})
    for key in ("market", "fundamentals", "news", "sentiment"):
        current.setdefault(key, [])
    current[agent_key] = findings
    return current


def format_specialist_findings_for_prompt(findings_by_agent: Any) -> str:
    """Render the blackboard into a compact, citation-friendly prompt block."""

    if not isinstance(findings_by_agent, dict):
        return "No structured specialist findings available."

    lines: list[str] = []
    for agent in ("market", "fundamentals", "news", "sentiment"):
        findings = findings_by_agent.get(agent) or []
        if not findings:
            continue
        lines.append(f"## {agent.capitalize()} findings")
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            finding_id = finding.get("id", f"{agent}-unknown")
            lines.append(
                "- {id} [{direction} / confidence:{confidence} / importance:{importance}] "
                "{claim}".format(
                    id=finding_id,
                    direction=finding.get("direction", "neutral"),
                    confidence=finding.get("confidence", "unknown"),
                    importance=finding.get("importance", "unknown"),
                    claim=finding.get("claim", ""),
                )
            )
            for evidence in finding.get("evidence", []) or []:
                if not isinstance(evidence, dict):
                    continue
                metric = evidence.get("metric") or evidence.get("source") or "evidence"
                value = f" = {evidence.get('value')}" if evidence.get("value") else ""
                date = f" ({evidence.get('date')})" if evidence.get("date") else ""
                detail = evidence.get("detail", "")
                lines.append(f"  - {metric}{value}{date}: {detail}")

    return "\n".join(lines) if lines else "No structured specialist findings available."
