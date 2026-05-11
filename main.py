"""
Climate Policy Debate: Multi-Agent System
Three agents debate a climate policy question:
  - Urban Environmentalist: ground-level city impacts and environmental justice
  - Investor/Finance: argues costs, ROI, economic feasibility
  - Decision Maker: weighs both sides and delivers a final verdict

Tracing: all LLM calls are automatically traced in Langfuse.
Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env to enable.
"""

from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langfuse.langchain import CallbackHandler as LangfuseCallback
import operator

load_dotenv()

# ============================================================================
# STATE
# ============================================================================

class DebateState(TypedDict):
    topic: str                          # The policy question being debated
    round: int                          # Current debate round (1 or 2)
    scientist_argument: str             # Climate scientist's position
    investor_argument: str              # Investor's position
    scientist_rebuttal: str             # Scientist responds to investor
    investor_rebuttal: str              # Investor responds to scientist
    decision: str                       # Decision maker's final verdict
    errors: Annotated[list, operator.add]


# ============================================================================
# LLM
# ============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# ============================================================================
# AGENTS
# ============================================================================

def climate_scientist(state: DebateState) -> dict:
    """Opens the debate from an urban environmentalist perspective."""
    try:
        topic = state["topic"]
        prompt = f"""You are Jordan Rivera, a passionate urban environmentalist and community organizer
based in a densely populated city that suffers from air pollution, urban heat islands, and
flooding from extreme weather events.
You are debating the following climate policy question:

"{topic}"

Make a compelling opening argument FOR strong climate action on this topic.
Draw on:
- Ground-level impacts on city residents: air quality, heat, flooding, health
- Environmental justice — low-income and minority communities hit hardest
- Green infrastructure opportunities: parks, transit, clean energy in cities
- The urgent need for local and national policy to protect urban communities

Be direct, speak from lived urban experience, and keep it under 200 words.
"""
        result = llm.invoke(prompt).content
        print(f"\n[Urban Environmentalist] Argument ready ({len(result)} chars)")
        return {"scientist_argument": result, "errors": []}
    except Exception as e:
        print(f"[Urban Environmentalist] FAILED: {e}")
        return {"scientist_argument": "", "errors": [f"Scientist error: {e}"]}


def investor(state: DebateState) -> dict:
    """Responds with economic and financial concerns."""
    try:
        topic = state["topic"]
        scientist_arg = state["scientist_argument"]
        prompt = f"""You are Marcus Reid, a managing partner at a major infrastructure investment fund.
You are debating the following climate policy question:

"{topic}"

The urban environmentalist just argued:
---
{scientist_arg}
---

Now make a counter-argument representing the financial and economic perspective.
Consider:
- Implementation costs and economic disruption
- Jobs and industry transition risks
- Return on investment and feasibility timelines
- Market-driven vs. policy-driven solutions

You are NOT a climate denier — acknowledge the science — but push back on the
pace, scale, or method of the proposed policy. Under 200 words.
"""
        result = llm.invoke(prompt).content
        print(f"[Investor] Argument ready ({len(result)} chars)")
        return {"investor_argument": result, "errors": []}
    except Exception as e:
        print(f"[Investor] FAILED: {e}")
        return {"investor_argument": "", "errors": [f"Investor error: {e}"]}


def scientist_rebuts(state: DebateState) -> dict:
    """Climate scientist fires back at the investor's concerns."""
    try:
        topic = state["topic"]
        investor_arg = state["investor_argument"]
        prompt = f"""You are Jordan Rivera, urban environmentalist.

The investor just responded to your argument on:
"{topic}"

Their argument:
---
{investor_arg}
---

Deliver a sharp, evidence-based rebuttal. Address their economic concerns directly,
show where the data disagrees, and reinforce why delay is more costly.
Under 150 words.
"""
        result = llm.invoke(prompt).content
        print(f"[Scientist Rebuttal] Ready ({len(result)} chars)")
        return {"scientist_rebuttal": result, "errors": []}
    except Exception as e:
        print(f"[Scientist Rebuttal] FAILED: {e}")
        return {"scientist_rebuttal": "", "errors": [f"Scientist rebuttal error: {e}"]}


def investor_rebuts(state: DebateState) -> dict:
    """Investor fires back at the scientist's rebuttal."""
    try:
        topic = state["topic"]
        scientist_rebuttal = state["scientist_rebuttal"]
        prompt = f"""You are Marcus Reid, infrastructure investor.

The urban environmentalist just rebutted your argument on:
"{topic}"

Their rebuttal:
---
{scientist_rebuttal}
---

Give your final counter. Acknowledge valid points, but make a clear case for your
preferred approach — whether that's phased implementation, market incentives, or
private-sector leadership over regulation. Under 150 words.
"""
        result = llm.invoke(prompt).content
        print(f"[Investor Rebuttal] Ready ({len(result)} chars)")
        return {"investor_rebuttal": result, "errors": []}
    except Exception as e:
        print(f"[Investor Rebuttal] FAILED: {e}")
        return {"investor_rebuttal": "", "errors": [f"Investor rebuttal error: {e}"]}


def decision_maker(state: DebateState) -> dict:
    """Synthesizes the debate and issues a final policy recommendation."""
    try:
        topic = state["topic"]
        prompt = f"""You are Minister Aiko Tanaka, a senior government official responsible
for climate and economic policy.

You have just witnessed a full debate on:
"{topic}"

== OPENING ARGUMENTS ==

URBAN ENVIRONMENTALIST:
{state["scientist_argument"]}

INVESTOR:
{state["investor_argument"]}

== REBUTTALS ==

URBAN ENVIRONMENTALIST (rebuttal):
{state["scientist_rebuttal"]}

INVESTOR (rebuttal):
{state["investor_rebuttal"]}

== YOUR TASK ==
Deliver a structured final decision. Include:
1. **Summary**: What each side got right
2. **Key tension**: The core trade-off you had to resolve
3. **Decision**: Your policy recommendation (be specific — approve, reject, or modify)
4. **Conditions**: Any caveats, timelines, or safeguards on your decision
5. **Rationale**: Why this decision is the right balance

Be decisive. Under 300 words.
"""
        result = llm.invoke(prompt).content
        print(f"[Decision Maker] Verdict ready ({len(result)} chars)")
        return {"decision": result, "errors": []}
    except Exception as e:
        print(f"[Decision Maker] FAILED: {e}")
        return {"decision": "", "errors": [f"Decision maker error: {e}"]}


# ============================================================================
# BUILD THE GRAPH
# ============================================================================

def build_debate_graph():
    graph = StateGraph(DebateState)

    graph.add_node("climate_scientist", climate_scientist)
    graph.add_node("investor", investor)
    graph.add_node("scientist_rebuttal", scientist_rebuts)
    graph.add_node("investor_rebuttal", investor_rebuts)
    graph.add_node("decision_maker", decision_maker)

    graph.set_entry_point("climate_scientist")
    graph.add_edge("climate_scientist", "investor")
    graph.add_edge("investor", "scientist_rebuttal")
    graph.add_edge("scientist_rebuttal", "investor_rebuttal")
    graph.add_edge("investor_rebuttal", "decision_maker")
    graph.add_edge("decision_maker", END)

    return graph.compile()


# ============================================================================
# RUNNER
# ============================================================================

def run_debate(topic: str) -> DebateState:
    graph = build_debate_graph()

    initial_state: DebateState = {
        "topic": topic,
        "round": 1,
        "scientist_argument": "",
        "investor_argument": "",
        "scientist_rebuttal": "",
        "investor_rebuttal": "",
        "decision": "",
        "errors": [],
    }

    print("\n" + "=" * 70)
    print(f"DEBATE TOPIC: {topic}")
    print("=" * 70)

    langfuse_handler = LangfuseCallback()
    result = graph.invoke(initial_state, config={"callbacks": [langfuse_handler]})

    print("\n" + "=" * 70)
    print("FULL DEBATE TRANSCRIPT")
    print("=" * 70)

    sections = [
        ("URBAN ENVIRONMENTALIST — Opening Argument", result["scientist_argument"]),
        ("INVESTOR — Opening Argument",               result["investor_argument"]),
        ("URBAN ENVIRONMENTALIST — Rebuttal",         result["scientist_rebuttal"]),
        ("INVESTOR — Rebuttal",                   result["investor_rebuttal"]),
        ("DECISION MAKER — Final Verdict",        result["decision"]),
    ]

    for title, content in sections:
        print(f"\n--- {title} ---")
        print(content)

    if result["errors"]:
        print(f"\n[ERRORS] {result['errors']}")

    langfuse_handler.flush()
    print(f"\n[Langfuse] Trace: https://cloud.langfuse.com/trace/{langfuse_handler.last_trace_id}")

    return result


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    topics = [
        "Given a 4 × 4 city grid and a limited number of trees, where should we plant trees to achieve the best balance between cooling, health, equity, biodiversity, and cost?",
        "Should carbon taxes replace all existing climate regulations?",
        "Should rich nations fund a $1 trillion climate transition fund for developing countries?",
    ]

    # Run the first debate topic by default; swap index to try others
    run_debate(topics[0])
