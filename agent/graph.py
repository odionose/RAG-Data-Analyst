import os
from typing import Annotated, TypedDict
from urllib import response
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.tools import execute_sql, search_vector_db
from agent.cost_tracker import CostTracker

load_dotenv()


# ─── State Definition ─────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    cost_tracker: CostTracker


# ─── Tools & LLM ──────────────────────────────────────────────────────────────
TOOLS = [execute_sql, search_vector_db]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  # review
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = """You are a financial research analyst assistant with access to two tools:

1. execute_sql: Query structured numerical data (revenue, profit, employee counts, etc.)
   Use this for precise numbers and comparisons across companies/years.

2. search_vector_db: Semantically search document text for qualitative information.
   Use this for strategy, risk factors, management commentary, business descriptions.

For complex questions, use BOTH tools to give a complete answer.
Always cite your sources (company name, year, document section).
When you have enough information, provide a clear, structured answer.
"""


# ─── Node Functions ───────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> AgentState:
    """
    The reasoning node. Gemini looks at all messages so far and decides:
    - Call a tool (returns a ToolCall message)
    - Generate a final answer (returns an AIMessage with no tool calls)
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)


    # Track token costs from Gemini's response metadata
    usage = response.usage_metadata
    if usage:
        state["cost_tracker"].record_loop(
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
    )

    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    Router: after agent node, decide what to do next.
    - Tool calls present → go to tools node
    - No tool calls → END (agent has final answer)
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ─── Build the Graph ──────────────────────────────────────────────────────────

def build_graph():
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", ToolNode(TOOLS))

    graph_builder.set_entry_point("agent")

    graph_builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END}
    )
    # After tools: always go back to agent (the ReAct loop)
    graph_builder.add_edge("tools", "agent")

    return graph_builder.compile()


# ─── Public interface ─────────────────────────────────────────────────────────

def run_agent(user_query: str) -> dict:
    graph = build_graph()
    tracker = CostTracker()

    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "cost_tracker": tracker,
    }

    final_state = graph.invoke(initial_state, config={"recursion_limit": 20})
    final_answer = final_state["messages"][-1].content
    cost_summary = tracker.summary()

    print(f"\n[COST SUMMARY] {cost_summary}")
    return {"answer": final_answer, "cost": cost_summary}


if __name__ == "__main__":
    result = run_agent("What is Amazon's stance on cybersecurity?")
    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print(result["answer"])
    print("\nCOST:", result["cost"])