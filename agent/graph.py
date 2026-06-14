import os
from typing import Annotated, TypedDict, Any

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.errors import GraphRecursionError

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
    model="gemini-2.5-flash-lite",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

llm_with_tools = llm.bind_tools(TOOLS)


SYSTEM_PROMPT = """You are a financial research analyst assistant with access to two tools:

1. execute_sql:
   Query structured numerical data such as revenue, profit, employees, R&D,
   assets, and operating margin.

2. search_vector_db:
   Search qualitative company document text such as strategy, risks,
   cybersecurity, management commentary, competition, and business descriptions.

Tool-use rules:
- Use execute_sql for precise numbers and comparisons.
- Use search_vector_db for qualitative document commentary.
- For complex questions, use both tools when useful.

Error-handling rules:
- If a tool returns SQL TOOL ERROR, VECTOR TOOL ERROR, or TOOL RUNTIME ERROR,
  do not blindly repeat the same failed call.
- You may fix a bad SQL query once.
- If one tool is unavailable, use the other tool if it helps.
- If evidence is incomplete, clearly say what failed and answer only from
  available evidence.
- Do not loop forever trying the same failing tool.

Answer rules:
- Always cite available sources using company name, year, and document section
  when that information is available.
- Give a clear, structured answer.
- If the tools fail, explain the failure clearly instead of crashing.
"""


# ─── Error Formatting ─────────────────────────────────────────────────────────

def format_tool_error(error: Exception) -> str:
    """
    Converts unexpected tool crashes into a message the LLM can reason over.
    This prevents ToolNode from crashing the whole graph.
    """
    return (
        f"TOOL RUNTIME ERROR [{type(error).__name__}]: {error}\n"
        "Do not repeat the same failed tool call. Fix the input once, try another "
        "tool if useful, or explain the limitation to the user."
    )


# ─── Node Functions ───────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> AgentState:
    """
    Reasoning node.

    Gemini decides whether to:
    - call a tool, or
    - produce a final answer.

    This node is wrapped so LLM/cost-tracking failures do not crash the graph.
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    tracker = state.get("cost_tracker")

    try:
        response = llm_with_tools.invoke(messages)

    except Exception as error:
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"AGENT ERROR [{type(error).__name__}]: The LLM call failed.\n\n"
                        f"Details: {error}\n\n"
                        "Check GOOGLE_API_KEY, Gemini model access, quota, network "
                        "connectivity, and package versions."
                    )
                )
            ],
            "cost_tracker": tracker,
        }

    # Track token costs safely.
    usage: dict[str, Any] = getattr(response, "usage_metadata", None) or {}

    if tracker and usage:
        try:
            tracker.record_loop(
                input_tokens=usage.get("input_tokens", 0) or 0,
                output_tokens=usage.get("output_tokens", 0) or 0,
            )
        except Exception:
            # Cost tracking should never break the agent.
            pass

    return {
        "messages": [response],
        "cost_tracker": tracker,
    }


def should_continue(state: AgentState) -> str:
    """
    Router:
    - If the latest AI message contains tool calls, execute tools.
    - Otherwise, stop.
    """
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return END


# ─── Build the Graph ──────────────────────────────────────────────────────────

def build_graph():
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("agent", agent_node)

    # Explicit tool error handling is important.
    # Without this, a thrown tool exception can crash the whole graph.
    graph_builder.add_node(
        "tools",
        ToolNode(
            TOOLS,
            handle_tool_errors=format_tool_error,
        ),
    )

    graph_builder.set_entry_point("agent")

    graph_builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )

    # ReAct loop: tools return results, then agent reasons again.
    graph_builder.add_edge("tools", "agent")

    return graph_builder.compile()


# ─── Public Interface ─────────────────────────────────────────────────────────

def run_agent(user_query: str) -> dict:
    graph = build_graph()
    tracker = CostTracker()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_query)],
        "cost_tracker": tracker,
    }

    try:
        final_state = graph.invoke(
            initial_state,
            config={"recursion_limit": 20},
        )

        final_message = final_state["messages"][-1]
        final_answer = getattr(final_message, "content", "") or ""

        if not final_answer.strip():
            final_answer = (
                "AGENT ERROR [EmptyFinalAnswer]: The graph ended without a final "
                "text answer. This usually means the model produced unresolved tool "
                "calls or stopped unexpectedly."
            )

    except GraphRecursionError as error:
        final_answer = (
            "AGENT ERROR [GraphRecursionError]: The agent hit the recursion limit "
            "before reaching a final answer.\n\n"
            "This usually means it kept cycling between the agent and tools without "
            "settling on a response. Check whether a tool is repeatedly failing or "
            "returning unclear output.\n\n"
            f"Details: {error}"
        )

    except Exception as error:
        final_answer = (
            f"AGENT ERROR [{type(error).__name__}]: The graph failed unexpectedly.\n\n"
            f"Details: {error}"
        )

    try:
        cost_summary = tracker.summary()
    except Exception as error:
        cost_summary = {
            "error": f"Cost summary failed: {type(error).__name__}: {error}"
        }

    print(f"\n[COST SUMMARY] {cost_summary}")

    return {
        "answer": final_answer,
        "cost": cost_summary,
    }


if __name__ == "__main__":
    result = run_agent("What is Amazon's stance on cybersecurity?")

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print(result["answer"])
    print("\nCOST:", result["cost"])