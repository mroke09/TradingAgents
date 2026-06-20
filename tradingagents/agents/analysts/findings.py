"""Structured finding extraction for specialist analyst reports."""

from __future__ import annotations

import logging
import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from tradingagents.agents.schemas import (
    SpecialistAnalysisOutput,
    SpecialistFinding,
    SpecialistFindingsReport,
    VerifiedFact,
)
from tradingagents.agents.utils.structured import bind_structured

logger = logging.getLogger(__name__)

SpecialistKey = Literal["market", "fundamentals", "news", "sentiment"]
SUBMIT_SPECIALIST_ANALYSIS_TOOL_NAME = "submit_specialist_analysis"


@dataclass(frozen=True)
class SpecialistAnalysisResult:
    report: str
    findings: list[dict[str, Any]]
    facts: list[dict[str, Any]]
    message: AIMessage


@tool(args_schema=SpecialistAnalysisOutput)
def submit_specialist_analysis(
    markdown_report: str,
    findings: list[dict[str, Any]],
) -> str:
    """Submit the final specialist report and structured findings.

    Call this exactly once when no more data tools are needed. The application
    intercepts the tool call and saves the report/findings; the tool itself is
    not executed by a data ToolNode.
    """

    return "specialist analysis submitted"


def specialist_final_tool_instruction(agent_key: SpecialistKey) -> str:
    return (
        "\n\nWhen your specialist analysis is complete, do not answer as plain "
        "text. Call `submit_specialist_analysis` exactly once. Put the full "
        "readable markdown report in `markdown_report`, and put 3-8 auditable "
        "structured findings in `findings`. Every finding must be grounded in "
        "the markdown report and tool/data evidence. Each finding's `agent` "
        f"must be `{agent_key}`, and each finding id must start with "
        f"`{agent_key}-`."
    )


def bind_findings_extractor(llm: Any, agent_name: str) -> Any | None:
    return bind_structured(llm, SpecialistFindingsReport, f"{agent_name} Findings")


def bind_specialist_analysis_output(llm: Any, agent_name: str) -> Any | None:
    return bind_structured(llm, SpecialistAnalysisOutput, f"{agent_name} Output")


def finalize_specialist_analysis_from_message(
    message: Any,
    *,
    agent_key: SpecialistKey,
    trade_date: str,
    deterministic_facts: list[dict[str, Any]] | None = None,
) -> SpecialistAnalysisResult | None:
    """Return final analysis from a final tool call or final text response.

    A return value of ``None`` means the model requested ordinary data tools and
    the analyst loop should continue.
    """

    for tool_call in getattr(message, "tool_calls", None) or []:
        if _tool_call_name(tool_call) != SUBMIT_SPECIALIST_ANALYSIS_TOOL_NAME:
            continue
        try:
            output = _analysis_output_from_payload(_tool_call_args(tool_call))
        except Exception as exc:
            logger.warning(
                "%s specialist final submission was malformed (%s); "
                "falling back to message content",
                agent_key,
                exc,
            )
            return _result_from_report_text(
                str(getattr(message, "content", "") or ""),
                agent_key=agent_key,
                trade_date=trade_date,
                facts=deterministic_facts,
            )
        return _result_from_output(
            output,
            agent_key=agent_key,
            trade_date=trade_date,
            facts=deterministic_facts,
        )

    if getattr(message, "tool_calls", None):
        return None

    content = str(getattr(message, "content", "") or "")
    try:
        output = _analysis_output_from_content(content)
    except Exception:
        return _result_from_report_text(
            content,
            agent_key=agent_key,
            trade_date=trade_date,
            facts=deterministic_facts,
        )
    return _result_from_output(
        output,
        agent_key=agent_key,
        trade_date=trade_date,
        facts=deterministic_facts,
    )


def invoke_specialist_analysis_output(
    structured_llm: Any | None,
    plain_llm: Any,
    prompt: Any,
    *,
    agent_key: SpecialistKey,
    agent_name: str,
    trade_date: str,
    deterministic_facts: list[dict[str, Any]] | None = None,
) -> SpecialistAnalysisResult:
    """Invoke one specialist call that returns markdown and findings together."""

    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            output = (
                result
                if isinstance(result, SpecialistAnalysisOutput)
                else SpecialistAnalysisOutput.model_validate(result)
            )
            return _result_from_output(
                output,
                agent_key=agent_key,
                trade_date=trade_date,
                facts=deterministic_facts,
            )
        except Exception as exc:
            logger.warning(
                "%s: structured specialist output failed (%s); retrying as free text",
                agent_name,
                exc,
            )

    response = plain_llm.invoke(prompt)
    return _result_from_report_text(
        str(getattr(response, "content", "") or ""),
        agent_key=agent_key,
        trade_date=trade_date,
        facts=deterministic_facts,
    )


def extract_verified_facts_from_messages(
    messages: list[Any],
    *,
    agent_key: SpecialistKey,
    trade_date: str,
) -> list[dict[str, Any]]:
    """Extract deterministic facts from trusted tool outputs in the message history."""

    facts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for message in messages or []:
        source_tool = _message_tool_name(message)
        content = str(getattr(message, "content", "") or "")
        if source_tool != "get_verified_market_snapshot" and (
            "Verified market data snapshot" not in content
        ):
            continue
        for fact in _facts_from_verified_market_snapshot(
            content,
            agent_key=agent_key,
            trade_date=trade_date,
            source_tool="get_verified_market_snapshot",
        ):
            if fact["id"] in seen_ids:
                continue
            seen_ids.add(fact["id"])
            facts.append(fact)
    return facts


def update_deterministic_facts(
    state: dict[str, Any],
    agent_key: SpecialistKey,
    facts: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    current = dict(state.get("deterministic_facts") or {})
    for key in ("market", "fundamentals", "news", "sentiment"):
        current.setdefault(key, [])
    current[agent_key] = facts
    return current


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


def _tool_call_name(tool_call: Any) -> str | None:
    if isinstance(tool_call, dict):
        return tool_call.get("name") or (tool_call.get("function") or {}).get("name")
    return getattr(tool_call, "name", None)


def _tool_call_args(tool_call: Any) -> Any:
    if isinstance(tool_call, dict):
        args = tool_call.get("args")
        if args is None:
            args = (tool_call.get("function") or {}).get("arguments")
    else:
        args = getattr(tool_call, "args", None)
    if isinstance(args, str):
        return json.loads(args)
    return args or {}


def _analysis_output_from_payload(payload: Any) -> SpecialistAnalysisOutput:
    if isinstance(payload, SpecialistAnalysisOutput):
        return payload
    if isinstance(payload, str):
        return SpecialistAnalysisOutput.model_validate_json(payload)
    return SpecialistAnalysisOutput.model_validate(payload)


def _analysis_output_from_content(content: str) -> SpecialistAnalysisOutput:
    text = content.strip()
    if not text:
        raise ValueError("empty specialist output")
    try:
        return SpecialistAnalysisOutput.model_validate_json(text)
    except Exception:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if not match:
        raise ValueError("no specialist JSON object found")
    return SpecialistAnalysisOutput.model_validate_json(match.group(1))


def _result_from_output(
    output: SpecialistAnalysisOutput,
    *,
    agent_key: SpecialistKey,
    trade_date: str,
    facts: list[dict[str, Any]] | None = None,
) -> SpecialistAnalysisResult:
    report = output.markdown_report.strip()
    deterministic_facts = list(facts or [])
    findings = [
        _normalize_finding(finding, agent_key, trade_date, index)
        for index, finding in enumerate(output.findings, start=1)
    ]
    findings = _attach_fact_ids_to_findings(findings, deterministic_facts)
    return SpecialistAnalysisResult(
        report=report,
        findings=findings,
        facts=deterministic_facts,
        message=AIMessage(content=report),
    )


def _result_from_report_text(
    report: str,
    *,
    agent_key: SpecialistKey,
    trade_date: str,
    facts: list[dict[str, Any]] | None = None,
) -> SpecialistAnalysisResult:
    return SpecialistAnalysisResult(
        report=report.strip(),
        findings=[],
        facts=list(facts or []),
        message=AIMessage(content=report.strip()),
    )


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


def _message_tool_name(message: Any) -> str:
    name = getattr(message, "name", None)
    if name:
        return str(name)
    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    return str(additional_kwargs.get("name") or "")


def _facts_from_verified_market_snapshot(
    content: str,
    *,
    agent_key: SpecialistKey,
    trade_date: str,
    source_tool: str,
) -> list[dict[str, Any]]:
    latest_date = _extract_line_value(content, "- Latest trading row used:")
    section = None
    facts: list[dict[str, Any]] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line.startswith("### Latest verified OHLCV row"):
            section = "ohlcv"
            continue
        if line.startswith("### Verified technical indicators"):
            section = "indicator"
            continue
        if line.startswith("### Recent verified closes"):
            section = "recent"
            continue
        if not line.startswith("|") or "---" in line or "Field" in line or "Indicator" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2 or section not in {"ohlcv", "indicator"}:
            continue
        metric, value = cells[0], cells[1]
        if not metric or not value or value.startswith("N/A"):
            continue
        fact_id = "{agent}-fact-{metric}".format(
            agent=agent_key,
            metric=_slug(metric),
        )
        fact = VerifiedFact(
            id=fact_id,
            agent=agent_key,
            source_tool=source_tool,
            metric=metric,
            value=value,
            unit=None,
            date=latest_date or trade_date,
            vendor="yfinance",
            freshness="latest trading row on or before analysis date",
        )
        facts.append(fact.model_dump(mode="json"))
    return facts


def _extract_line_value(content: str, prefix: str) -> str | None:
    for line in content.splitlines():
        if line.strip().startswith(prefix):
            return line.split(prefix, 1)[1].strip()
    return None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "value"


def _attach_fact_ids_to_findings(
    findings: list[dict[str, Any]],
    facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not facts:
        return findings
    by_metric = {
        str(fact.get("metric", "")).lower(): fact
        for fact in facts
        if fact.get("metric") and fact.get("id")
    }
    for finding in findings:
        for evidence in finding.get("evidence", []) or []:
            if not isinstance(evidence, dict) or evidence.get("fact_id"):
                continue
            metric = str(evidence.get("metric") or "").lower()
            source = str(evidence.get("source") or "").lower()
            fact = by_metric.get(metric)
            if fact is None and "verified_market_snapshot" in source:
                value = str(evidence.get("value") or "").strip()
                fact = next(
                    (
                        candidate
                        for candidate in facts
                        if value and value in str(candidate.get("value", ""))
                    ),
                    None,
                )
            if fact is not None:
                evidence["fact_id"] = fact["id"]
                evidence.setdefault("source", fact.get("source_tool"))
                evidence.setdefault("date", fact.get("date"))
    return findings


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
