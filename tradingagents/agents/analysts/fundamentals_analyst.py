from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.findings import (
    finalize_specialist_analysis_from_message,
    specialist_final_tool_instruction,
    submit_specialist_analysis,
    update_specialist_findings,
)
from tradingagents.agents.skills import render_configured_skill_prompt
from tradingagents.agents.utils.agent_utils import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_instrument_context_from_state,
    get_language_instruction,
)


def create_fundamentals_analyst(llm, config=None):
    language_instruction = get_language_instruction(config)
    tools = [
        get_fundamentals,
        get_balance_sheet,
        get_cashflow,
        get_income_statement,
    ]
    llm_tools = tools + [submit_specialist_analysis]
    tool_names = ", ".join([tool.name for tool in llm_tools])
    default_system_message = (
        "You are a researcher tasked with analyzing fundamental information over the past week about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
        + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
        + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements."
        + language_instruction
        + specialist_final_tool_instruction("fundamentals")
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
    ).partial(tool_names=tool_names)
    chain = prompt | llm.bind_tools(llm_tools)

    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)
        ticker = state.get("company_of_interest", "")
        asset_type = state.get("asset_type", "stock")

        skill_message = render_configured_skill_prompt(
            config=config,
            agent_id="fundamentals_analyst",
            context={
                "ticker": ticker,
                "trade_date": current_date,
                "current_date": current_date,
                "asset_type": asset_type,
                "instrument_context": instrument_context,
                "tool_names": tool_names,
            },
        )
        system_message = (
            skill_message
            + language_instruction
            + specialist_final_tool_instruction("fundamentals")
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
            agent_key="fundamentals",
            trade_date=current_date,
        )
        if analysis is None:
            return {
                "messages": [result],
                "fundamentals_report": "",
            }

        update = {
            "messages": [analysis.message],
            "fundamentals_report": analysis.report,
        }
        if analysis.report:
            update["specialist_findings"] = update_specialist_findings(
                state, "fundamentals", analysis.findings
            )
        return update

    return fundamentals_analyst_node
