"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

import json

from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.skills import render_configured_skill_prompt
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_research_manager(llm, config=None):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        history = state["investment_debate_state"].get("history", "")

        investment_debate_state = state["investment_debate_state"]
        structured_arguments = json.dumps(
            {
                "bull_arguments": investment_debate_state.get("bull_arguments", []),
                "bear_arguments": investment_debate_state.get("bear_arguments", []),
            },
            ensure_ascii=False,
            indent=2,
        )
        context = {
            "instrument_context": instrument_context,
            "ticker": state.get("company_of_interest", ""),
            "trade_date": state.get("trade_date", ""),
            "asset_type": state.get("asset_type", "stock"),
            "structured_arguments": structured_arguments,
            "history": history,
        }
        default_prompt = f"""As the Research Manager and debate facilitator, your role is to critically evaluate this round of debate and deliver a clear, actionable investment plan for the trader.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction in the bull thesis; recommend taking or growing the position
- **Overweight**: Constructive view; recommend gradually increasing exposure
- **Hold**: Balanced view; recommend maintaining the current position
- **Underweight**: Cautious view; recommend trimming exposure
- **Sell**: Strong conviction in the bear thesis; recommend exiting or avoiding the position

Commit to a clear stance whenever the debate's strongest arguments warrant one; reserve Hold for situations where the evidence on both sides is genuinely balanced.

---

You are a judge, not a transcript summarizer. Identify which evidence is strongest, which side carried the debate, what the real disagreement is, and what remains unresolved. Prefer structured argument evidence with specialist finding ids when available.

**Structured Bull/Bear Arguments:**
{structured_arguments}

---

**Debate History:**
{history}"""
        prompt_body = render_configured_skill_prompt(
            config=config,
            agent_id="research_manager",
            context=context,
        ) or default_prompt
        prompt = prompt_body + get_language_instruction()

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "bull_arguments": investment_debate_state.get("bull_arguments", []),
            "bear_arguments": investment_debate_state.get("bear_arguments", []),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
