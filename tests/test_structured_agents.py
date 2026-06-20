"""Tests for structured-output agents (Trader, Research Manager, Sentiment Analyst).

The Portfolio Manager has its own coverage in tests/test_memory_log.py
(which exercises the full memory-log → PM injection cycle).  This file
covers the parallel schemas, render functions, and graceful-fallback
behavior we added for the Trader, Research Manager, and Sentiment Analyst
so they share the same deterministic output shape.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from tradingagents.agents.analysts.sentiment_analyst import create_sentiment_analyst
from tradingagents.agents.analysts.findings import (
    extract_specialist_findings,
    format_specialist_findings_for_prompt,
)
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.schemas import (
    DebateEvidence,
    FindingEvidence,
    PortfolioRating,
    ResearchPlan,
    ResearcherArgument,
    SentimentBand,
    SentimentReport,
    SpecialistFinding,
    SpecialistFindingsReport,
    TraderAction,
    TraderProposal,
    render_researcher_argument,
    render_research_plan,
    render_sentiment_report,
    render_trader_proposal,
)
from tradingagents.agents.trader.trader import create_trader

# ---------------------------------------------------------------------------
# Render functions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderTraderProposal:
    def test_minimal_required_fields(self):
        p = TraderProposal(action=TraderAction.HOLD, reasoning="Balanced setup; no edge.")
        md = render_trader_proposal(p)
        assert "**Action**: Hold" in md
        assert "**Reasoning**: Balanced setup; no edge." in md
        # The trailing FINAL TRANSACTION PROPOSAL line is preserved for the
        # analyst stop-signal text and any external code that greps for it.
        assert "FINAL TRANSACTION PROPOSAL: **HOLD**" in md

    def test_optional_fields_included_when_present(self):
        p = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Strong technicals + fundamentals.",
            entry_price=189.5,
            stop_loss=178.0,
            position_sizing="6% of portfolio",
        )
        md = render_trader_proposal(p)
        assert "**Action**: Buy" in md
        assert "**Entry Price**: 189.5" in md
        assert "**Stop Loss**: 178.0" in md
        assert "**Position Sizing**: 6% of portfolio" in md
        assert "FINAL TRANSACTION PROPOSAL: **BUY**" in md

    def test_optional_fields_omitted_when_absent(self):
        p = TraderProposal(action=TraderAction.SELL, reasoning="Guidance cut.")
        md = render_trader_proposal(p)
        assert "Entry Price" not in md
        assert "Stop Loss" not in md
        assert "Position Sizing" not in md
        assert "FINAL TRANSACTION PROPOSAL: **SELL**" in md


@pytest.mark.unit
class TestRenderResearchPlan:
    def test_required_fields(self):
        p = ResearchPlan(
            recommendation=PortfolioRating.OVERWEIGHT,
            rationale="Bull case carried; tailwinds intact.",
            strategic_actions="Build position over two weeks; cap at 5%.",
        )
        md = render_research_plan(p)
        assert "**Recommendation**: Overweight" in md
        assert "**Rationale**: Bull case carried" in md
        assert "**Strategic Actions**: Build position" in md

    def test_all_5_tier_ratings_render(self):
        for rating in PortfolioRating:
            p = ResearchPlan(
                recommendation=rating,
                rationale="r",
                strategic_actions="s",
            )
            md = render_research_plan(p)
            assert f"**Recommendation**: {rating.value}" in md

    def test_manager_judgment_fields_render_when_present(self):
        p = ResearchPlan(
            recommendation=PortfolioRating.BUY,
            rationale="Bull evidence is stronger.",
            strategic_actions="Build position gradually.",
            winning_side="bull",
            key_disagreement="Whether margin expansion can persist.",
            strongest_bull_evidence=["fundamentals-1: margins improving"],
            strongest_bear_evidence=["market-2: overbought technicals"],
            unresolved_questions=["Next quarter gross margin guide"],
            confidence="medium",
        )
        md = render_research_plan(p)
        assert "**Winning Side**: bull" in md
        assert "**Key Disagreement**: Whether margin expansion can persist." in md
        assert "- fundamentals-1: margins improving" in md
        assert "**Manager Confidence**: Medium" in md


@pytest.mark.unit
class TestSpecialistFindings:
    def test_schema_accepts_auditable_finding(self):
        finding = SpecialistFinding(
            id="market-1",
            agent="market",
            category="trend",
            claim="The trend remains constructive.",
            evidence=[
                FindingEvidence(
                    source="get_verified_market_snapshot",
                    metric="close_50_sma",
                    value="above price",
                    date="2026-01-15",
                    detail="The report says price is above the 50-day average.",
                )
            ],
            direction="bullish",
            confidence="medium",
            importance="high",
            as_of_date="2026-01-15",
        )

        assert finding.id == "market-1"
        assert finding.evidence[0].source == "get_verified_market_snapshot"

    def test_extract_specialist_findings_normalizes_agent_id_and_date(self):
        structured = MagicMock()
        structured.invoke.return_value = SpecialistFindingsReport(
            findings=[
                SpecialistFinding(
                    id="1",
                    agent="market",
                    category="trend",
                    claim="Trend is constructive.",
                    evidence=[
                        FindingEvidence(
                            source="report",
                            metric="trend",
                            value="positive",
                            detail="Report cites constructive price action.",
                        )
                    ],
                    direction="bullish",
                    confidence="high",
                    importance="high",
                )
            ]
        )

        findings = extract_specialist_findings(
            structured,
            agent_key="market",
            report_text="Trend is constructive.",
            trade_date="2026-01-15",
            instrument_context="NVDA is NVIDIA.",
        )

        assert findings[0]["id"] == "market-1"
        assert findings[0]["as_of_date"] == "2026-01-15"

    def test_format_findings_for_prompt_includes_ids(self):
        text = format_specialist_findings_for_prompt(
            {
                "market": [
                    {
                        "id": "market-1",
                        "direction": "bullish",
                        "confidence": "high",
                        "importance": "medium",
                        "claim": "Trend is constructive.",
                        "evidence": [
                            {"source": "report", "metric": "RSI", "value": "58", "detail": "Healthy momentum."}
                        ],
                    }
                ]
            }
        )

        assert "market-1" in text
        assert "RSI = 58" in text


@pytest.mark.unit
class TestRenderResearcherArgument:
    def test_required_sections(self):
        argument = ResearcherArgument(
            thesis="Growth and margins support upside.",
            key_points=["Demand remains strong", "Margins are resilient"],
            evidence=[
                DebateEvidence(
                    finding_id="fundamentals-1",
                    source="fundamentals",
                    claim="Revenue growth",
                    detail="The fundamentals report shows growth remains positive.",
                    importance="high",
                )
            ],
            rebuttal="Valuation risk is real but supported by growth.",
            assumptions=["Demand does not collapse"],
            confidence="medium",
        )

        md = render_researcher_argument(argument)

        assert "**Thesis**: Growth and margins support upside." in md
        assert "- Demand remains strong" in md
        assert "[fundamentals fundamentals-1 / high] Revenue growth" in md
        assert "**Rebuttal**: Valuation risk" in md
        assert "**Confidence**: Medium" in md


# ---------------------------------------------------------------------------
# Trader agent: structured happy path + fallback
# ---------------------------------------------------------------------------


def _make_trader_state():
    return {
        "company_of_interest": "NVDA",
        "investment_plan": "**Recommendation**: Buy\n**Rationale**: ...\n**Strategic Actions**: ...",
    }


def _structured_trader_llm(captured: dict, proposal: TraderProposal | None = None):
    """Build a MagicMock LLM whose with_structured_output binding captures the
    prompt and returns a real TraderProposal so render_trader_proposal works.
    """
    if proposal is None:
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Strong setup.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or proposal
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestTraderAgent:
    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="AI capex cycle intact; institutional flows constructive.",
            entry_price=189.5,
            stop_loss=178.0,
            position_sizing="6% of portfolio",
        )
        llm = _structured_trader_llm(captured, proposal)
        trader = create_trader(llm)
        result = trader(_make_trader_state())
        plan = result["trader_investment_plan"]
        assert "**Action**: Buy" in plan
        assert "**Entry Price**: 189.5" in plan
        assert "FINAL TRANSACTION PROPOSAL: **BUY**" in plan
        # The same rendered markdown is also added to messages for downstream agents.
        assert plan in result["messages"][0].content

    def test_prompt_includes_investment_plan(self):
        captured = {}
        llm = _structured_trader_llm(captured)
        trader = create_trader(llm)
        trader(_make_trader_state())
        # The investment plan is in the user message of the captured prompt.
        prompt = captured["prompt"]
        assert any("Proposed Investment Plan" in m["content"] for m in prompt)

    def test_falls_back_to_freetext_when_structured_unavailable(self):
        plain_response = (
            "**Action**: Sell\n\nGuidance cut hits margins.\n\n"
            "FINAL TRANSACTION PROPOSAL: **SELL**"
        )
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain_response)
        trader = create_trader(llm)
        result = trader(_make_trader_state())
        assert result["trader_investment_plan"] == plain_response


# ---------------------------------------------------------------------------
# Research Manager agent: structured happy path + fallback
# ---------------------------------------------------------------------------


def _make_rm_state():
    return {
        "company_of_interest": "NVDA",
        "investment_debate_state": {
            "history": "Bull and bear arguments here.",
            "bull_history": "Bull says...",
            "bear_history": "Bear says...",
            "current_response": "",
            "judge_decision": "",
            "count": 1,
        },
    }


def _structured_rm_llm(captured: dict, plan: ResearchPlan | None = None):
    if plan is None:
        plan = ResearchPlan(
            recommendation=PortfolioRating.HOLD,
            rationale="Balanced view across both sides.",
            strategic_actions="Hold current position; reassess after earnings.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or plan
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestResearchManagerAgent:
    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        plan = ResearchPlan(
            recommendation=PortfolioRating.OVERWEIGHT,
            rationale="Bull case is stronger; AI tailwind intact.",
            strategic_actions="Build position gradually over two weeks.",
        )
        llm = _structured_rm_llm(captured, plan)
        rm = create_research_manager(llm)
        result = rm(_make_rm_state())
        ip = result["investment_plan"]
        assert "**Recommendation**: Overweight" in ip
        assert "**Rationale**: Bull case" in ip
        assert "**Strategic Actions**: Build position" in ip

    def test_prompt_uses_5_tier_rating_scale(self):
        """The RM prompt must list all five tiers so the schema enum matches user expectations."""
        captured = {}
        llm = _structured_rm_llm(captured)
        rm = create_research_manager(llm)
        rm(_make_rm_state())
        prompt = captured["prompt"]
        for tier in ("Buy", "Overweight", "Hold", "Underweight", "Sell"):
            assert f"**{tier}**" in prompt, f"missing {tier} in prompt"

    def test_falls_back_to_freetext_when_structured_unavailable(self):
        plain_response = "**Recommendation**: Sell\n\n**Rationale**: ...\n\n**Strategic Actions**: ..."
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain_response)
        rm = create_research_manager(llm)
        result = rm(_make_rm_state())
        assert result["investment_plan"] == plain_response


# ---------------------------------------------------------------------------
# Bull / Bear researchers: structured arguments + fallback
# ---------------------------------------------------------------------------


def _make_researcher_state():
    return {
        "company_of_interest": "NVDA",
        "asset_type": "stock",
        "trade_date": "2026-01-15",
        "instrument_context": "NVDA is NVIDIA Corporation.",
        "market_report": "Market trend is constructive.",
        "sentiment_report": "Sentiment is bullish.",
        "news_report": "Recent news is positive.",
        "fundamentals_report": "Revenue growth remains strong.",
        "specialist_findings": {
            "fundamentals": [
                {
                    "id": "fundamentals-1",
                    "agent": "fundamentals",
                    "category": "growth",
                    "claim": "Revenue growth remains strong.",
                    "direction": "bullish",
                    "confidence": "high",
                    "importance": "high",
                    "evidence": [
                        {
                            "source": "fundamentals_report",
                            "metric": "revenue",
                            "detail": "Revenue growth remains strong.",
                        }
                    ],
                }
            ]
        },
        "investment_debate_state": {
            "history": "",
            "bull_history": "",
            "bear_history": "",
            "bull_arguments": [],
            "bear_arguments": [],
            "current_response": "",
            "judge_decision": "",
            "count": 0,
        },
    }


def _researcher_argument():
    return ResearcherArgument(
        thesis="Strong revenue growth supports the constructive case.",
        key_points=["Revenue growth is intact", "Sentiment is constructive"],
        evidence=[
            DebateEvidence(
                finding_id="fundamentals-1",
                source="fundamentals",
                claim="Revenue growth",
                detail="The fundamentals report says revenue growth remains strong.",
                importance="high",
            )
        ],
        rebuttal="Valuation risk matters, but the growth evidence offsets it.",
        assumptions=["Demand remains durable"],
        confidence="high",
    )


def _structured_researcher_llm(captured: dict, argument: ResearcherArgument | None = None):
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or (argument or _researcher_argument())
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestResearcherAgents:
    def test_bull_researcher_stores_structured_argument_and_markdown_history(self):
        captured = {}
        bull = create_bull_researcher(_structured_researcher_llm(captured))

        debate = bull(_make_researcher_state())["investment_debate_state"]

        assert debate["current_response"].startswith("Bull Analyst:")
        assert "**Thesis**: Strong revenue growth" in debate["bull_history"]
        assert debate["bull_arguments"][0]["thesis"].startswith("Strong revenue growth")
        assert debate["bull_arguments"][0]["evidence"][0]["source"] == "fundamentals"
        assert debate["bull_arguments"][0]["evidence"][0]["finding_id"] == "fundamentals-1"
        assert "Structured specialist findings" in captured["prompt"]
        assert "fundamentals-1" in captured["prompt"]
        assert "Bear Counterpoints" in captured["prompt"]

    def test_bear_researcher_stores_structured_argument_and_markdown_history(self):
        captured = {}
        bear = create_bear_researcher(_structured_researcher_llm(captured))

        debate = bear(_make_researcher_state())["investment_debate_state"]

        assert debate["current_response"].startswith("Bear Analyst:")
        assert "**Thesis**: Strong revenue growth" in debate["bear_history"]
        assert debate["bear_arguments"][0]["confidence"] == "high"
        assert "Bull Counterpoints" in captured["prompt"]

    def test_researcher_falls_back_to_free_text_without_structured_data(self):
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content="Free-text bull argument.")
        bull = create_bull_researcher(llm)

        debate = bull(_make_researcher_state())["investment_debate_state"]

        assert "Bull Analyst: Free-text bull argument." in debate["bull_history"]
        assert debate["bull_arguments"] == []


# ---------------------------------------------------------------------------
# Sentiment Analyst: schema, render, structured happy path + fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderSentimentReport:
    def test_header_contains_band_and_score(self):
        report = SentimentReport(
            overall_band=SentimentBand.BULLISH,
            overall_score=7.2,
            confidence="high",
            narrative="Source breakdown here.",
        )
        md = render_sentiment_report(report)
        assert "**Overall Sentiment:** **Bullish**" in md
        assert "(Score: 7.2/10)" in md

    def test_header_contains_confidence(self):
        report = SentimentReport(
            overall_band=SentimentBand.NEUTRAL,
            overall_score=5.0,
            confidence="low",
            narrative="Limited data.",
        )
        assert "**Confidence:** Low" in render_sentiment_report(report)

    def test_narrative_preserved_in_output(self):
        narrative = "## Breakdown\n\nStockTwits: 70% bullish.\n\n| Signal | Direction |\n|---|---|\n| News | Neutral |"
        report = SentimentReport(
            overall_band=SentimentBand.MILDLY_BULLISH,
            overall_score=6.0,
            confidence="medium",
            narrative=narrative,
        )
        assert narrative in render_sentiment_report(report)

    def test_all_six_bands_render(self):
        for band in SentimentBand:
            report = SentimentReport(
                overall_band=band, overall_score=5.0,
                confidence="medium", narrative="n",
            )
            assert band.value in render_sentiment_report(report)

    def test_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            SentimentReport(
                overall_band=SentimentBand.BULLISH, overall_score=11.0,
                confidence="high", narrative="n",
            )


def _make_sentiment_state():
    return {
        "company_of_interest": "NVDA",
        "trade_date": "2026-01-15",
        "asset_type": "stock",
        "messages": [],
    }


def _structured_sentiment_llm(
    captured: dict,
    report: SentimentReport | None = None,
    findings: SpecialistFindingsReport | None = None,
):
    """MagicMock LLM whose structured binding captures the prompt and returns
    a real SentimentReport so render_sentiment_report works."""
    if report is None:
        report = SentimentReport(
            overall_band=SentimentBand.BULLISH, overall_score=7.5,
            confidence="high",
            narrative="StockTwits 75% bullish. News constructive. Reddit upbeat.",
        )
    if findings is None:
        findings = SpecialistFindingsReport(findings=[])
    sentiment_structured = MagicMock()
    sentiment_structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or report
    )
    findings_structured = MagicMock()
    findings_structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("findings_prompt", prompt) or findings
    )
    llm = MagicMock()
    llm.with_structured_output.side_effect = lambda schema: (
        sentiment_structured if schema is SentimentReport else findings_structured
    )
    return llm


@pytest.mark.unit
class TestSentimentAnalystAgent:
    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        report = SentimentReport(
            overall_band=SentimentBand.MILDLY_BEARISH, overall_score=4.0,
            confidence="medium", narrative="Mixed signals across sources.",
        )
        analyst = create_sentiment_analyst(_structured_sentiment_llm(captured, report))
        sr = analyst(_make_sentiment_state())["sentiment_report"]
        assert "**Overall Sentiment:** **Mildly Bearish**" in sr
        assert "(Score: 4.0/10)" in sr
        assert "Mixed signals across sources." in sr

    def test_sentiment_report_also_in_messages(self):
        captured = {}
        analyst = create_sentiment_analyst(_structured_sentiment_llm(captured))
        result = analyst(_make_sentiment_state())
        assert len(result["messages"]) == 1
        assert result["sentiment_report"] == result["messages"][0].content

    def test_sentiment_findings_are_added_to_blackboard(self):
        captured = {}
        findings = SpecialistFindingsReport(
            findings=[
                SpecialistFinding(
                    id="sentiment-1",
                    agent="sentiment",
                    category="retail_sentiment",
                    claim="Retail sentiment is bullish.",
                    evidence=[
                        FindingEvidence(
                            source="StockTwits",
                            metric="bullish ratio",
                            value="75%",
                            detail="StockTwits messages skew bullish.",
                        )
                    ],
                    direction="bullish",
                    confidence="high",
                    importance="medium",
                )
            ]
        )
        analyst = create_sentiment_analyst(_structured_sentiment_llm(captured, findings=findings))
        result = analyst(_make_sentiment_state())
        assert result["specialist_findings"]["sentiment"][0]["id"] == "sentiment-1"

    def test_prompt_contains_ticker(self):
        captured = {}
        create_sentiment_analyst(_structured_sentiment_llm(captured))(_make_sentiment_state())
        assert any("NVDA" in str(m) for m in captured["prompt"])

    def test_falls_back_to_freetext_when_structured_unavailable(self):
        plain = "**Overall Sentiment:** **Bearish** (Score: 3.0/10)\n**Confidence:** Low\n\nLimited data."
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain)
        assert create_sentiment_analyst(llm)(_make_sentiment_state())["sentiment_report"] == plain

    def test_falls_back_to_freetext_when_structured_call_fails(self):
        plain = "Fallback free-text sentiment."
        structured = MagicMock()
        structured.invoke.side_effect = ValueError("bad JSON from model")
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        llm.invoke.return_value = MagicMock(content=plain)
        assert create_sentiment_analyst(llm)(_make_sentiment_state())["sentiment_report"] == plain
