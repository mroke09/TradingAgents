from tradingagents.agents.analysts.findings import format_specialist_findings_for_prompt
from tradingagents.agents.schemas import ResearcherArgument
from tradingagents.agents.skills import configured_skill_prompt
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import bind_structured

from .researcher_argument import invoke_researcher_argument, validate_argument_citations


def create_bear_researcher(llm, config=None):
    structured_llm = bind_structured(llm, ResearcherArgument, "Bear Researcher")
    language_instruction = get_language_instruction(config)

    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        specialist_findings = format_specialist_findings_for_prompt(
            state.get("specialist_findings", {})
        )
        instrument_context = get_instrument_context_from_state(state)
        asset_type = state.get("asset_type", "stock")
        target_label = "stock" if asset_type == "stock" else "asset"
        fundamentals_label = (
            "Company fundamentals report"
            if asset_type == "stock"
            else "Asset fundamentals report (may be unavailable for crypto)"
        )
        context = {
            "instrument_context": instrument_context,
            "asset_type": asset_type,
            "ticker": state.get("company_of_interest", ""),
            "trade_date": state.get("trade_date", ""),
            "market_research_report": market_research_report,
            "sentiment_report": sentiment_report,
            "news_report": news_report,
            "fundamentals_report": fundamentals_report,
            "specialist_findings": specialist_findings,
            "history": history,
            "current_response": current_response,
        }
        default_prompt = f"""You are a Bear Analyst making the case against investing in the {target_label}. Your goal is to present a well-reasoned argument emphasizing risks, challenges, and negative indicators. Leverage the provided research and data to highlight potential downsides and counter bullish arguments effectively.

Key points to focus on:

- Risks and Challenges: Highlight factors like market saturation, financial instability, or macroeconomic threats that could hinder the stock's performance.
- Competitive Weaknesses: Emphasize vulnerabilities such as weaker market positioning, declining innovation, or threats from competitors.
- Negative Indicators: Use evidence from financial data, market trends, or recent adverse news to support your position.
- Bull Counterpoints: Critically analyze the bull argument with specific data and sound reasoning, exposing weaknesses or over-optimistic assumptions.
- Engagement: Present your argument in a conversational style, directly engaging with the bull analyst's points and debating effectively rather than simply listing facts.

Resources available:

{instrument_context}
Structured specialist findings (primary evidence; cite finding ids in your evidence when applicable):
{specialist_findings}

Citation rules:
- Every evidence item grounded in the specialist findings should include its exact finding_id.
- Use source="other" only for synthesized cross-report reasoning that cannot cite one finding.
- Do not invent finding ids; cite only ids shown in the structured specialist findings block.

Fallback analyst reports:
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
{fundamentals_label}: {fundamentals_report}
Conversation history of the debate: {history}
Last bull argument: {current_response}
Use this information to deliver a compelling bear argument, refute the bull's claims, and engage in a dynamic debate that demonstrates the risks and weaknesses of investing in the {target_label}.
"""
        skill_template = configured_skill_prompt(config, "bear_researcher")
        prompt_body = skill_template.format(**context) if skill_template else default_prompt
        if skill_template and "{specialist_findings}" not in skill_template and specialist_findings:
            prompt_body += (
                "\n\nStructured specialist findings (primary evidence; cite "
                "finding ids in your evidence when applicable):\n"
                + specialist_findings
            )
        prompt = prompt_body + language_instruction

        argument_content, structured_argument = invoke_researcher_argument(
            structured_llm=structured_llm,
            plain_llm=llm,
            prompt=prompt,
            agent_name="Bear Researcher",
        )
        argument = f"Bear Analyst: {argument_content}"
        bear_arguments = list(investment_debate_state.get("bear_arguments", []))
        if structured_argument is not None:
            structured_argument = validate_argument_citations(
                structured_argument,
                state.get("specialist_findings", {}),
            )
            bear_arguments.append(structured_argument)

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "bull_arguments": investment_debate_state.get("bull_arguments", []),
            "bear_arguments": bear_arguments,
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
