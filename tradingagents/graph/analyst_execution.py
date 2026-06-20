from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from time import monotonic
from typing import Any

from tradingagents.dataflows.config import get_config, run_with_config


@dataclass(frozen=True)
class AnalystNodeSpec:
    key: str
    agent_node: str
    clear_node: str
    tool_node: str
    report_key: str


@dataclass(frozen=True)
class AnalystExecutionPlan:
    specs: list[AnalystNodeSpec]
    concurrency_limit: int

    @property
    def uses_parallel_execution(self) -> bool:
        return self.concurrency_limit > 1 and len(self.specs) > 1


ANALYST_NODE_SPECS: dict[str, AnalystNodeSpec] = {
    "market": AnalystNodeSpec(
        key="market",
        agent_node="Market Analyst",
        clear_node="Msg Clear Market",
        tool_node="tools_market",
        report_key="market_report",
    ),
    "social": AnalystNodeSpec(
        # Wire key stays "social" for saved-config back-compat; the
        # user-facing label is "Sentiment Analyst" to match the rename
        # that landed in v0.2.5 (sentiment_analyst now ingests news +
        # StockTwits + Reddit, not just social media).
        key="social",
        agent_node="Sentiment Analyst",
        clear_node="Msg Clear Sentiment",
        tool_node="tools_social",
        report_key="sentiment_report",
    ),
    "news": AnalystNodeSpec(
        key="news",
        agent_node="News Analyst",
        clear_node="Msg Clear News",
        tool_node="tools_news",
        report_key="news_report",
    ),
    "fundamentals": AnalystNodeSpec(
        key="fundamentals",
        agent_node="Fundamentals Analyst",
        clear_node="Msg Clear Fundamentals",
        tool_node="tools_fundamentals",
        report_key="fundamentals_report",
    ),
}


def build_analyst_execution_plan(
    selected_analysts: Iterable[str],
    concurrency_limit: int = 1,
) -> AnalystExecutionPlan:
    if concurrency_limit < 1:
        raise ValueError("analyst concurrency limit must be >= 1")

    specs: list[AnalystNodeSpec] = []
    for analyst_key in selected_analysts:
        spec = ANALYST_NODE_SPECS.get(analyst_key)
        if spec is None:
            raise ValueError(f"unknown analyst key: {analyst_key}")
        specs.append(spec)

    if not specs:
        raise ValueError("at least one analyst must be selected")

    return AnalystExecutionPlan(specs=specs, concurrency_limit=concurrency_limit)


def get_initial_analyst_node(plan: AnalystExecutionPlan) -> str:
    return plan.specs[0].agent_node


def create_parallel_analysts_node(
    plan: AnalystExecutionPlan,
    analyst_nodes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
    tool_nodes: dict[str, Any],
    *,
    max_iterations: int = 50,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a graph node that runs selected analysts concurrently.

    Each analyst gets an isolated copy of the incoming state. This keeps
    tool-calling transcripts separate: ToolNode reads the last AI message on the
    local ``messages`` list, so sharing that list across concurrent analysts can
    route one analyst's tool call to another analyst's tool node.
    """

    def parallel_analysts_node(state: dict[str, Any]) -> dict[str, Any]:
        reports: dict[str, Any] = {}
        active_config = get_config()
        max_workers = min(plan.concurrency_limit, len(plan.specs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_with_config,
                    active_config,
                    run_isolated_analyst,
                    spec,
                    analyst_nodes[spec.key],
                    tool_nodes[spec.key],
                    state,
                    max_iterations=max_iterations,
                ): spec
                for spec in plan.specs
            }
            for future in as_completed(futures):
                merge_parallel_result(reports, future.result())
        return reports

    return parallel_analysts_node


def run_isolated_analyst(
    spec: AnalystNodeSpec,
    analyst_node: Callable[[dict[str, Any]], dict[str, Any]],
    tool_node: Any,
    state: dict[str, Any],
    *,
    max_iterations: int = 50,
) -> dict[str, Any]:
    """Run one analyst and its tool loop against an isolated state copy."""

    local_state = copy_analyst_state(state)
    for _ in range(max_iterations):
        analyst_result = analyst_node(local_state)
        merge_local_result(local_state, analyst_result)

        last_message = last_local_message(analyst_result, local_state)
        if has_tool_calls(last_message):
            tool_result = tool_node.invoke(local_state)
            merge_local_result(local_state, tool_result)
            continue

        result = {spec.report_key: local_state.get(spec.report_key, "")}
        if "specialist_findings" in local_state:
            result["specialist_findings"] = local_state["specialist_findings"]
        return result

    raise RuntimeError(
        f"{spec.agent_node} exceeded {max_iterations} isolated tool iterations"
    )


def copy_analyst_state(state: dict[str, Any]) -> dict[str, Any]:
    local_state = dict(state)
    local_state["messages"] = list(state.get("messages", []))
    return local_state


def merge_local_result(
    local_state: dict[str, Any],
    result: dict[str, Any] | None,
) -> None:
    if not result:
        return
    for key, value in result.items():
        if key == "messages":
            local_state["messages"] = list(local_state.get("messages", [])) + list(value)
        else:
            local_state[key] = value


def merge_parallel_result(target: dict[str, Any], result: dict[str, Any] | None) -> None:
    if not result:
        return
    for key, value in result.items():
        if key in {"specialist_findings", "deterministic_facts"}:
            merged = dict(target.get("specialist_findings") or {})
            if key == "deterministic_facts":
                merged = dict(target.get("deterministic_facts") or {})
            if isinstance(value, dict):
                for agent, findings in value.items():
                    if findings:
                        merged[agent] = findings
                    else:
                        merged.setdefault(agent, findings)
            target[key] = merged
        else:
            target[key] = value


def last_local_message(
    result: dict[str, Any] | None,
    local_state: dict[str, Any],
) -> Any:
    result_messages = (result or {}).get("messages") or []
    if result_messages:
        return result_messages[-1]
    messages = local_state.get("messages") or []
    return messages[-1] if messages else None


def has_tool_calls(message: Any) -> bool:
    return bool(getattr(message, "tool_calls", None))


class AnalystWallTimeTracker:
    def __init__(self, plan: AnalystExecutionPlan):
        self.plan = plan
        self._started_at: dict[str, float] = {}
        self._wall_times: dict[str, float] = {}

    def mark_started(self, analyst_key: str, started_at: float | None = None) -> None:
        if analyst_key not in ANALYST_NODE_SPECS:
            raise ValueError(f"unknown analyst key: {analyst_key}")
        self._started_at.setdefault(analyst_key, monotonic() if started_at is None else started_at)

    def mark_completed(
        self,
        analyst_key: str,
        completed_at: float | None = None,
    ) -> None:
        if analyst_key not in ANALYST_NODE_SPECS:
            raise ValueError(f"unknown analyst key: {analyst_key}")
        if analyst_key in self._wall_times:
            return
        started_at = self._started_at.get(analyst_key)
        if started_at is None:
            return
        finished_at = monotonic() if completed_at is None else completed_at
        self._wall_times[analyst_key] = max(0.0, finished_at - started_at)

    def get_wall_times(self) -> dict[str, float]:
        return dict(self._wall_times)

    def format_summary(self) -> str:
        parts = []
        for spec in self.plan.specs:
            duration = self._wall_times.get(spec.key)
            if duration is not None:
                label = spec.agent_node.removesuffix(" Analyst")
                parts.append(f"{label} {duration:.2f}s")
        if not parts:
            return "Analyst wall time: pending"
        return "Analyst wall time: " + " | ".join(parts)


def sync_analyst_tracker_from_chunk(
    tracker: AnalystWallTimeTracker,
    chunk: dict[str, str],
    now: float | None = None,
) -> None:
    current_time = monotonic() if now is None else now
    active_found = False

    for spec in tracker.plan.specs:
        has_report = bool(chunk.get(spec.report_key))

        if has_report:
            tracker.mark_started(spec.key, started_at=current_time)
            tracker.mark_completed(spec.key, completed_at=current_time)
            continue

        if not active_found:
            tracker.mark_started(spec.key, started_at=current_time)
            active_found = True
