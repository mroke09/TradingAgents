from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.findings import (
    finalize_specialist_analysis_from_message,
    specialist_final_tool_instruction,
    submit_specialist_analysis,
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
    language_instruction = get_language_instruction(config)
    tools = [
        get_news,
        get_global_news,
        get_macro_indicators,
        get_prediction_markets,
    ]
    llm_tools = tools + [submit_specialist_analysis]
    tool_names = ", ".join([tool.name for tool in llm_tools])
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
    ).partial(tool_names=tool_names)
    chain = prompt | llm.bind_tools(llm_tools)

    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)

        default_system_message = (
            f"You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: get_news(query, start_date, end_date) for {asset_label}-specific or targeted news searches, get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news, get_macro_indicators(indicator, curr_date, look_back_days) to ground macro commentary in actual data from FRED (e.g. 'cpi', 'core_pce', 'unemployment', 'fed_funds_rate', '10y_treasury', 'yield_curve'), and get_prediction_markets(topic, limit) for live market-implied probabilities of forward-looking events (e.g. 'Fed rate cut', 'recession 2026', geopolitical or sector events). Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + language_instruction
            + specialist_final_tool_instruction("news")
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
                "tool_names": tool_names,
            },
        )
        system_message = (
            skill_message + language_instruction + specialist_final_tool_instruction("news")
            if skill_message
            else default_system_message
        )

        result = chain.invoke(
            {
                "messages": state["messages"],
                "system_message": system_message,
                "current_date": current_date,
                "instrument_context": instrument_context,
            }
        )

        analysis = finalize_specialist_analysis_from_message(
            result,
            agent_key="news",
            trade_date=current_date,
        )
        if analysis is None:
            return {
                "messages": [result],
                "news_report": "",
            }

        update = {
            "messages": [analysis.message],
            "news_report": analysis.report,
        }
        if analysis.report:
            update["specialist_findings"] = update_specialist_findings(
                state, "news", analysis.findings
            )
        return update

    return news_analyst_node
