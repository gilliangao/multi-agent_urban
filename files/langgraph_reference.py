"""
LANGGRAPH + LANGFUSE QUICK REFERENCE
For the hackathon—copy/paste snippets as needed
"""

# ============================================================================
# PATTERN 1: Add a new agent
# ============================================================================

def my_new_agent(state: ResearchState) -> dict:
    """Template for adding a new agent"""
    try:
        # 1. Extract what you need from state
        previous_result = state["previous_field"]
        
        # 2. Do your agent work (LLM call, API, logic, etc.)
        output = llm.invoke(f"Process: {previous_result}")
        
        # 3. Return updated state
        return {
            "my_output_field": output.content,
            "errors": state["errors"]  # Preserve error history
        }
    except Exception as e:
        return {
            "my_output_field": "",
            "errors": state["errors"] + [f"MyAgent error: {str(e)}"]
        }

# Then add to graph:
# graph.add_node("my_agent_name", my_new_agent)
# graph.add_edge("previous_node", "my_agent_name")


# ============================================================================
# PATTERN 2: Conditional routing (branching logic)
# ============================================================================

def should_validate(state: ResearchState):
    """Decide whether to proceed to validation"""
    quality = state["validation"].get("quality_score", 0)
    
    # Return the name of the next node
    if quality < 5:
        return "researcher"  # Loop back to researcher
    else:
        return "end"  # Done

# Add to graph:
# graph.add_conditional_edges("analyzer", should_validate)
# graph.add_edge("should_validate", "researcher")
# graph.add_edge("should_validate", END)


# ============================================================================
# PATTERN 3: Parallel agents (run multiple agents at once)
# ============================================================================

from langgraph.graph import StateGraph, END

def build_parallel_graph():
    """Agents that can run in parallel"""
    graph = StateGraph(ResearchState)
    
    graph.add_node("researcher", researcher_agent)
    graph.add_node("fact_checker", fact_checker_agent)  # Parallel with researcher
    
    # Both start at entry
    graph.set_entry_point("researcher")
    
    # But they can run simultaneously if structured right
    graph.add_node("merger", merge_parallel_results)
    graph.add_edge("researcher", "merger")
    graph.add_edge("fact_checker", "merger")
    
    return graph.compile()

def merge_parallel_results(state: ResearchState) -> dict:
    """Combine results from parallel agents"""
    return {
        "merged_output": state["research"] + "\n" + state.get("fact_check", ""),
        "errors": state["errors"]
    }


# ============================================================================
# PATTERN 4: Add evaluation metrics
# ============================================================================

def evaluate_with_custom_metrics(graph, test_cases):
    """Measure what matters for YOUR project"""
    results = []
    
    for test in test_cases:
        start = time.time()
        result = graph.invoke({...})
        elapsed = time.time() - start
        
        # Define your metrics
        metrics = {
            "latency": elapsed,
            "token_count": count_tokens(result),  # Depends on your LLM
            "coherence": measure_coherence(result["analysis"]),  # Custom scorer
            "errors": len(result["errors"]),
            "success": len(result["errors"]) == 0
        }
        results.append(metrics)
    
    # Aggregate
    return {
        "avg_latency": sum(r["latency"] for r in results) / len(results),
        "success_rate": sum(1 for r in results if r["success"]) / len(results),
        "avg_coherence": sum(r["coherence"] for r in results) / len(results)
    }


# ============================================================================
# PATTERN 5: LangFuse integration - detailed tracing
# ============================================================================

from langfuse.callback import CallbackHandler
from langfuse import Langfuse

def run_with_detailed_trace(graph, query: str):
    """Show judges exactly what your agents decided"""
    
    # Setup handler
    handler = CallbackHandler()
    
    # Run graph
    result = graph.invoke(
        {"query": query, ...},
        config={"callbacks": [handler]}
    )
    
    # Manually log agent decisions (makes trace prettier)
    client = Langfuse()
    trace = client.trace(
        name="research_pipeline",
        input={"query": query}
    )
    
    # Log each agent's output as a span
    trace.span(
        name="researcher",
        input={"query": query},
        output={"research": result["research"][:100] + "..."}
    )
    
    trace.span(
        name="analyzer",
        input={"research": result["research"][:100]},
        output={"analysis": result["analysis"][:100] + "..."}
    )
    
    trace.span(
        name="validator",
        input={"analysis": result["analysis"][:100]},
        output={"validation": result["validation"]}
    )
    
    return result


# ============================================================================
# PATTERN 6: Error handling & retry logic
# ============================================================================

from langchain.schema import BaseMessage
import time

def agent_with_retry(state: ResearchState, max_retries: int = 2) -> dict:
    """Automatically retry on failure"""
    
    for attempt in range(max_retries):
        try:
            result = llm.invoke("Your prompt here")
            return {"output": result.content, "errors": state["errors"]}
        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "output": "",
                    "errors": state["errors"] + [f"Failed after {max_retries} attempts: {str(e)}"]
                }
            time.sleep(2 ** attempt)  # Exponential backoff


# ============================================================================
# PATTERN 7: Add tool/API calls to agents
# ============================================================================

def researcher_with_tools(state: ResearchState) -> dict:
    """Agent that can call external APIs/tools"""
    
    # Define tools your agent can use
    tools = {
        "search_web": search_web,
        "fetch_paper": fetch_paper,
        "call_api": call_external_api
    }
    
    # Prompt agent to decide which tool to use
    response = llm.invoke(f"""
        You have access to:
        - search_web(query) - search the web
        - fetch_paper(title) - fetch research papers
        
        User query: {state["query"]}
        
        Decide which tool to use and return the tool_name and arguments.
    """)
    
    # Parse response and call tool
    tool_name = extract_tool_name(response)
    tool_args = extract_args(response)
    result = tools[tool_name](**tool_args)
    
    return {"research": result, "errors": state["errors"]}


# ============================================================================
# PATTERN 8: Save and load graphs
# ============================================================================

import pickle

def save_graph(graph, filename: str):
    """Persist your graph for reuse"""
    with open(filename, "wb") as f:
        pickle.dump(graph, f)

def load_graph(filename: str):
    """Load a saved graph"""
    with open(filename, "rb") as f:
        return pickle.load(f)

# Use: save_graph(graph, "my_graph.pkl")


# ============================================================================
# LANGGRAPH DEBUGGING TIPS
# ============================================================================

"""
1. Print intermediate state:
   result = graph.invoke(initial_state)
   print("Research:", result["research"][:200])
   print("Analysis:", result["analysis"][:200])

2. Check what node is running:
   graph.get_graph().draw_ascii()
   # Shows the full graph structure visually

3. Step through graph manually:
   from langgraph.graph import END
   compiled = graph.compile()
   for event in compiled.stream(initial_state):
       print("Event:", event)  # See each step
       
4. Verify state schema:
   print(ResearchState.__annotations__)  # Check required fields

5. Log agent inputs/outputs:
   print(f"Agent received: {state}")
   print(f"Agent returning: {return_dict}")
"""


# ============================================================================
# HACKATHON CHECKLIST - BEFORE SUNDAY
# ============================================================================

"""
□ Can you define and run a 3-node graph?
□ Can you add a conditional edge that branches based on state?
□ Can you measure success metrics (latency, quality, errors)?
□ Can you run your graph on 5+ test cases?
□ Does your code handle errors gracefully?
□ Can you visualize your graph structure?
□ Have you set up LangFuse and seen a trace in the dashboard?
□ Can you modify agents and re-run in < 1 minute?
□ Do you have a skeleton ready for Monday morning?
□ Can you explain how your graph works in 30 seconds?
"""


# ============================================================================
# MONDAY MORNING STRATEGY
# ============================================================================

"""
09:00 - Arrive, register, find team
09:30 - Team brainstorm (10 min)
       - Agree on scientific problem
       - Sketch 3-4 agent roles
10:00 - START CODING
       - Copy this starter template
       - Modify agents for your problem
       - Plug in APIs/data sources
12:00 - FIRST WORKING PROTOTYPE
       - Get something end-to-end
       - Measure one metric
13:00 - Lunch break (rest, recharge)
14:00 - IMPROVE
       - Better prompts
       - More metrics
       - Add LangFuse
16:00 - OPTIMIZE & POLISH
       - Make sure evaluation is rigorous
       - Document decisions
17:00 - PREPARE PRESENTATION
       - Show graph architecture
       - Show eval results
       - Show LangFuse trace
"""

