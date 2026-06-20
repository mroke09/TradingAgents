from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.findings import (
    bind_findings_extractor,
    extract_specialist_findings,
    update_specialist_findings,
)
from tradingagents.agents.skills import render_configured_skill_prompt
from tradingagents.agents.utils.agent_utils import (
    get_global_news,
    get_instrument_context_from_state,
    get_language_instruction,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
)


def create_news_analyst(llm, config=None):
    findings_llm = bind_findings_extractor(llm, "News Analyst")

    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)

        tools = [
            get_news,
            get_global_news,
            get_macro_indicators,
            get_prediction_markets,
        ]

        default_system_message = (
            f"You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: get_news(query, start_date, end_date) for {asset_label}-specific or targeted news searches, get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news, get_macro_indicators(indicator, curr_date, look_back_days) to ground macro commentary in actual data from FRED (e.g. 'cpi', 'core_pce', 'unemployment', 'fed_funds_rate', '10y_treasury', 'yield_curve'), and get_prediction_markets(topic, limit) for live market-implied probabilities of forward-looking events (e.g. 'Fed rate cut', 'recession 2026', geopolitical or sector events). Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + get_language_instruction()
        )
        skill_message = render_configured_skill_prompt(
            config=config,
            agent_id="news_analyst",
            context={
                "ticker": state.get("company_of_interest", ""),
                "trade_date": current_date,
                "current_date": current_date,
                "asset_type": asset_type,
                "instrument_context": instrument_context,
                "tool_names": ", ".join([tool.name for tool in tools]),
            },
        )
        system_message = (
            skill_message + get_language_instruction()
            if skill_message
            else default_system_message
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        update = {
            "messages": [result],
            "news_report": report,
        }
        if report:
            findings = extract_specialist_findings(
                findings_llm,
                agent_key="news",
                report_text=report,
                trade_date=current_date,
                instrument_context=instrument_context,
            )
            update["specialist_findings"] = update_specialist_findings(
                state, "news", findings
            )
        return update

    return news_analyst_node
