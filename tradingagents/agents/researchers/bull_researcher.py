from tradingagents.agents.analysts.findings import format_specialist_findings_for_prompt
from tradingagents.agents.schemas import ResearcherArgument
from tradingagents.agents.skills import configured_skill_prompt
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import bind_structured

from .researcher_argument import invoke_researcher_argument


def create_bull_researcher(llm, config=None):
    structured_llm = bind_structured(llm, ResearcherArgument, "Bull Researcher")

    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

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
        default_prompt = f"""You are a Bull Analyst advocating for investing in the {target_label}. Your task is to build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators. Leverage the provided research and data to address concerns and counter bearish arguments effectively.

Key points to focus on:
- Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.
- Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.
- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.
- Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, addressing concerns thoroughly and showing why the bull perspective holds stronger merit.
- Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points and debating effectively rather than just listing data.

Resources available:
{instrument_context}
Structured specialist findings (primary evidence; cite finding ids in your evidence when applicable):
{specialist_findings}

Fallback analyst reports:
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
{fundamentals_label}: {fundamentals_report}
Conversation history of the debate: {history}
Last bear argument: {current_response}
Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position.
"""
        skill_template = configured_skill_prompt(config, "bull_researcher")
        prompt_body = skill_template.format(**context) if skill_template else default_prompt
        if skill_template and "{specialist_findings}" not in skill_template and specialist_findings:
            prompt_body += (
                "\n\nStructured specialist findings (primary evidence; cite "
                "finding ids in your evidence when applicable):\n"
                + specialist_findings
            )
        prompt = prompt_body + get_language_instruction()

        argument_content, structured_argument = invoke_researcher_argument(
            structured_llm=structured_llm,
            plain_llm=llm,
            prompt=prompt,
            agent_name="Bull Researcher",
        )
        argument = f"Bull Analyst: {argument_content}"
        bull_arguments = list(investment_debate_state.get("bull_arguments", []))
        if structured_argument is not None:
            bull_arguments.append(structured_argument)

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_arguments": bull_arguments,
            "bear_arguments": investment_debate_state.get("bear_arguments", []),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
