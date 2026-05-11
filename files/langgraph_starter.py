"""
LangGraph + LangFuse Starter: 3-Agent Research Pipeline for Hackathon
This is a working example you can run, modify, and extend immediately.
"""

import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import time

load_dotenv()

# ============================================================================
# 1. DEFINE STATE
# ============================================================================
class ResearchState(TypedDict):
    """State that flows through the graph"""
    query: str
    research: str
    analysis: str
    validation: dict
    errors: list


# ============================================================================
# 2. INITIALIZE LLM
# ============================================================================
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)


# ============================================================================
# 3. DEFINE AGENTS (Node Functions)
# ============================================================================

def researcher_agent(state: ResearchState) -> dict:
    """
    Agent 1: Research
    Takes a query and generates research findings.
    """
    try:
        query = state["query"]
        prompt = f"""You are a research assistant. 
        Research the following topic and provide key findings:
        
        Topic: {query}
        
        Format your response as:
        1. Main findings
        2. Key points
        3. Summary
        """
        
        research_result = llm.invoke(prompt).content
        print(f"✓ Researcher completed: {len(research_result)} chars")
        
        return {
            "research": research_result,
            "errors": []
        }
    except Exception as e:
        print(f"✗ Researcher failed: {str(e)}")
        return {
            "research": "",
            "errors": [f"Researcher error: {str(e)}"]
        }


def analyzer_agent(state: ResearchState) -> dict:
    """
    Agent 2: Analyze
    Takes research findings and extracts key insights.
    """
    try:
        research = state["research"]
        
        if not research:
            return {
                "analysis": "",
                "errors": state["errors"] + ["No research to analyze"]
            }
        
        prompt = f"""You are an analyst. Analyze the following research findings
        and extract the 3 most important insights:
        
        Research findings:
        {research}
        
        Provide your analysis in a structured format with exactly 3 insights.
        """
        
        analysis_result = llm.invoke(prompt).content
        print(f"✓ Analyzer completed: {len(analysis_result)} chars")
        
        return {
            "analysis": analysis_result,
            "errors": state["errors"]
        }
    except Exception as e:
        print(f"✗ Analyzer failed: {str(e)}")
        return {
            "analysis": "",
            "errors": state["errors"] + [f"Analyzer error: {str(e)}"]
        }


def validator_agent(state: ResearchState) -> dict:
    """
    Agent 3: Validate
    Checks the quality of analysis and flags issues.
    """
    try:
        analysis = state["analysis"]
        research = state["research"]
        
        if not analysis or not research:
            return {
                "validation": {
                    "is_valid": False,
                    "quality_score": 0,
                    "issues": ["Incomplete research or analysis"]
                },
                "errors": state["errors"] + ["Validation failed: missing inputs"]
            }
        
        prompt = f"""You are a quality validator. 
        Assess this analysis on a scale of 1-10 for:
        1. Accuracy
        2. Completeness
        3. Clarity
        
        Analysis to validate:
        {analysis}
        
        Return your assessment as:
        Overall Score: X/10
        Accuracy: X/10
        Completeness: X/10
        Clarity: X/10
        Issues found: [list]
        """
        
        validation_text = llm.invoke(prompt).content
        
        # Parse score (simple extraction)
        try:
            score = int(validation_text.split("Overall Score:")[1].split("/10")[0].strip())
        except:
            score = 5
        
        validation_result = {
            "is_valid": score >= 6,
            "quality_score": score,
            "details": validation_text
        }
        
        print(f"✓ Validator completed: quality_score={score}")
        
        return {
            "validation": validation_result,
            "errors": state["errors"]
        }
    except Exception as e:
        print(f"✗ Validator failed: {str(e)}")
        return {
            "validation": {
                "is_valid": False,
                "quality_score": 0,
                "error": str(e)
            },
            "errors": state["errors"] + [f"Validator error: {str(e)}"]
        }


# ============================================================================
# 4. BUILD THE GRAPH
# ============================================================================

def build_graph():
    """Construct the LangGraph state graph"""
    graph = StateGraph(ResearchState)
    
    # Add nodes (agents)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("analyzer", analyzer_agent)
    graph.add_node("validator", validator_agent)
    
    # Add edges (flow)
    graph.add_edge("researcher", "analyzer")
    graph.add_edge("analyzer", "validator")
    graph.add_edge("validator", END)
    
    # Set entry point
    graph.set_entry_point("researcher")
    
    return graph.compile()


# ============================================================================
# 5. EVALUATION FUNCTION
# ============================================================================

def evaluate_system(graph, test_cases: list) -> dict:
    """
    Run the graph on multiple test cases and measure performance.
    """
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test {i+1}/{len(test_cases)}: {test_case['query'][:50]}... ---")
        
        start_time = time.time()
        
        try:
            result = graph.invoke({
                "query": test_case["query"],
                "research": "",
                "analysis": "",
                "validation": {},
                "errors": []
            })
            
            elapsed = time.time() - start_time
            
            test_result = {
                "query": test_case["query"],
                "success": len(result["errors"]) == 0,
                "latency_sec": round(elapsed, 2),
                "quality_score": result.get("validation", {}).get("quality_score", 0),
                "errors": result["errors"]
            }
            
            results.append(test_result)
            
            print(f"  Success: {test_result['success']}")
            print(f"  Latency: {test_result['latency_sec']}s")
            print(f"  Quality: {test_result['quality_score']}/10")
            
        except Exception as e:
            results.append({
                "query": test_case["query"],
                "success": False,
                "latency_sec": time.time() - start_time,
                "error": str(e)
            })
            print(f"  FAILED: {str(e)}")
    
    # Summary statistics
    successful = sum(1 for r in results if r["success"])
    avg_latency = sum(r.get("latency_sec", 0) for r in results) / len(results)
    avg_quality = sum(r.get("quality_score", 0) for r in results if r.get("quality_score")) / max(1, sum(1 for r in results if r.get("quality_score")))
    
    summary = {
        "total_tests": len(results),
        "successful": successful,
        "success_rate": round(successful / len(results) * 100, 1),
        "avg_latency_sec": round(avg_latency, 2),
        "avg_quality_score": round(avg_quality, 1),
        "individual_results": results
    }
    
    return summary


# ============================================================================
# 6. LANGFUSE INTEGRATION (Optional but recommended for judges!)
# ============================================================================

def run_with_langfuse(graph, query: str):
    """
    Run the graph with LangFuse tracing.
    This creates a beautiful trace visible in the LangFuse dashboard.
    """
    try:
        from langfuse.callback import CallbackHandler
        
        handler = CallbackHandler()
        
        result = graph.invoke(
            {
                "query": query,
                "research": "",
                "analysis": "",
                "validation": {},
                "errors": []
            },
            config={"callbacks": [handler]}
        )
        
        print("\n✓ Trace sent to LangFuse!")
        print("  Check your dashboard at: https://cloud.langfuse.com")
        return result
        
    except ImportError:
        print("⚠ LangFuse not installed. Run: pip install langfuse")
        return graph.invoke({
            "query": query,
            "research": "",
            "analysis": "",
            "validation": {},
            "errors": []
        })


# ============================================================================
# 7. MAIN - RUN THE SYSTEM
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph + LangFuse: 3-Agent Research Pipeline")
    print("=" * 60)
    
    # Build the graph
    print("\n[1/3] Building graph...")
    graph = build_graph()
    print("✓ Graph built successfully")
    
    # Define test cases
    print("\n[2/3] Running evaluation on 5 test cases...")
    test_cases = [
        {"query": "What is photosynthesis?"},
        {"query": "Compare machine learning and deep learning"},
        {"query": "Explain the water cycle"},
        {"query": "What are the benefits of renewable energy?"},
        {"query": "How does the immune system work?"}
    ]
    
    # Evaluate
    eval_results = evaluate_system(graph, test_cases)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total tests: {eval_results['total_tests']}")
    print(f"Success rate: {eval_results['success_rate']}%")
    print(f"Avg latency: {eval_results['avg_latency_sec']}s")
    print(f"Avg quality: {eval_results['avg_quality_score']}/10")
    
    # Run one example with tracing
    print("\n[3/3] Running single example with LangFuse tracing...")
    print("(Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env for full tracing)")
    example_result = run_with_langfuse(graph, "What is quantum computing?")
    
    print("\n" + "=" * 60)
    print("DONE! You're ready for the hackathon.")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Test this script: python langgraph_starter.py")
    print("2. Modify agents for your scientific problem")
    print("3. Add more complex evaluation metrics")
    print("4. Set up LangFuse for full traceability")
