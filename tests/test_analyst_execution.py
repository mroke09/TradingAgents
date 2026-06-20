import unittest
from threading import Barrier

from tradingagents.graph.analyst_execution import (
    AnalystWallTimeTracker,
    build_analyst_execution_plan,
    create_parallel_analysts_node,
    get_initial_analyst_node,
    sync_analyst_tracker_from_chunk,
)


class AnalystExecutionPlanTests(unittest.TestCase):
    def test_build_plan_preserves_selected_order(self):
        plan = build_analyst_execution_plan(["news", "market"], concurrency_limit=2)

        self.assertEqual([spec.key for spec in plan.specs], ["news", "market"])
        self.assertEqual(plan.concurrency_limit, 2)
        self.assertEqual(plan.specs[0].agent_node, "News Analyst")
        self.assertEqual(plan.specs[0].tool_node, "tools_news")
        self.assertEqual(plan.specs[0].clear_node, "Msg Clear News")

    def test_rejects_unknown_analyst_keys(self):
        with self.assertRaises(ValueError):
            build_analyst_execution_plan(["market", "macro"])

    def test_requires_positive_concurrency_limit(self):
        with self.assertRaises(ValueError):
            build_analyst_execution_plan(["market"], concurrency_limit=0)

    def test_get_initial_analyst_node_uses_plan_metadata(self):
        plan = build_analyst_execution_plan(["fundamentals", "news"])

        self.assertEqual(
            get_initial_analyst_node(plan),
            "Fundamentals Analyst",
        )

    def test_parallel_execution_requires_multiple_analysts_and_workers(self):
        self.assertFalse(
            build_analyst_execution_plan(["market"], concurrency_limit=4).uses_parallel_execution
        )
        self.assertFalse(
            build_analyst_execution_plan(["market", "news"], concurrency_limit=1).uses_parallel_execution
        )
        self.assertTrue(
            build_analyst_execution_plan(["market", "news"], concurrency_limit=2).uses_parallel_execution
        )

    def test_social_key_displays_as_sentiment_analyst(self):
        # The wire key stays "social" for saved-config back-compat, but the
        # user-visible agent_node label must match the v0.2.5 rename so the
        # wall-time summary and any future consumer of agent_node says
        # "Sentiment Analyst" rather than the legacy "Social Analyst".
        plan = build_analyst_execution_plan(["social"])
        spec = plan.specs[0]
        self.assertEqual(spec.key, "social")
        self.assertEqual(spec.agent_node, "Sentiment Analyst")
        self.assertEqual(spec.report_key, "sentiment_report")


class ParallelAnalystNodeTests(unittest.TestCase):
    def test_runs_analysts_concurrently_and_merges_reports(self):
        plan = build_analyst_execution_plan(["market", "news"], concurrency_limit=2)
        barrier = Barrier(2)

        def make_analyst(report_key, report_text):
            def analyst(_state):
                barrier.wait(timeout=1)
                return {
                    "messages": [FakeMessage(report_text)],
                    report_key: report_text,
                }

            return analyst

        node = create_parallel_analysts_node(
            plan,
            {
                "market": make_analyst("market_report", "market done"),
                "news": make_analyst("news_report", "news done"),
            },
            {
                "market": FakeToolNode(),
                "news": FakeToolNode(),
            },
        )

        result = node({"messages": [FakeMessage("start")]})

        self.assertEqual(result["market_report"], "market done")
        self.assertEqual(result["news_report"], "news done")

    def test_keeps_tool_loop_isolated_per_analyst(self):
        plan = build_analyst_execution_plan(["market"], concurrency_limit=2)
        calls = {"analyst": 0, "tool": 0}

        def analyst(state):
            calls["analyst"] += 1
            if calls["analyst"] == 1:
                return {"messages": [FakeMessage("need tool", tool_calls=[{"id": "call-1"}])]}
            return {
                "messages": [FakeMessage("final")],
                "market_report": "saw " + state["messages"][-1].content,
            }

        node = create_parallel_analysts_node(
            plan,
            {"market": analyst},
            {"market": FakeToolNode(calls)},
        )

        result = node({"messages": [FakeMessage("start")]})

        self.assertEqual(result["market_report"], "saw tool result")
        self.assertEqual(calls, {"analyst": 2, "tool": 1})


class AnalystWallTimeTrackerTests(unittest.TestCase):
    def test_records_wall_time_when_analyst_completes(self):
        plan = build_analyst_execution_plan(["market", "news"])
        tracker = AnalystWallTimeTracker(plan)

        tracker.mark_started("market", started_at=10.0)
        tracker.mark_completed("market", completed_at=13.5)

        self.assertEqual(tracker.get_wall_times(), {"market": 3.5})

    def test_formats_summary_in_plan_order(self):
        plan = build_analyst_execution_plan(["news", "market"])
        tracker = AnalystWallTimeTracker(plan)

        tracker.mark_started("market", started_at=20.0)
        tracker.mark_completed("market", completed_at=22.25)
        tracker.mark_started("news", started_at=10.0)
        tracker.mark_completed("news", completed_at=14.0)

        self.assertEqual(
            tracker.format_summary(),
            "Analyst wall time: News 4.00s | Market 2.25s",
        )

    def test_syncs_wall_time_from_sequential_chunks(self):
        plan = build_analyst_execution_plan(["market", "news"])
        tracker = AnalystWallTimeTracker(plan)

        sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)
        self.assertEqual(tracker.get_wall_times(), {})

        sync_analyst_tracker_from_chunk(
            tracker,
            {"market_report": "done"},
            now=13.0,
        )
        self.assertEqual(tracker.get_wall_times(), {"market": 3.0})

        sync_analyst_tracker_from_chunk(
            tracker,
            {"market_report": "done", "news_report": "done"},
            now=18.0,
        )
        self.assertEqual(
            tracker.get_wall_times(),
            {"market": 3.0, "news": 5.0},
        )


class FakeMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeToolNode:
    def __init__(self, calls=None):
        self.calls = calls

    def invoke(self, _state):
        if self.calls is not None:
            self.calls["tool"] += 1
        return {"messages": [FakeMessage("tool result")]}
