"""
PROJECT IDEAS FOR MULTI-AGENT SYSTEMS IN SCIENTIFIC RESEARCH
Tailored for the hackathon - use these to brainstorm with your team Monday morning
"""

# ============================================================================
# IDEA 1: LITERATURE REVIEW AUTOMATION
# ============================================================================

PROJECT_1 = """
Name: Automated Literature Review Pipeline
Domain: Any research field

Problem: Scientists spend weeks manually reading and synthesizing papers.

Multi-Agent Solution:
  Agent 1 (Searcher):
    - Takes research question
    - Formulates search queries
    - Retrieves relevant paper titles & abstracts
    
  Agent 2 (Analyst):
    - Reads abstracts
    - Extracts methodology, findings, limitations
    - Categorizes papers by relevance
    
  Agent 3 (Synthesizer):
    - Combines findings across papers
    - Identifies consensus and contradictions
    - Generates a synthesis report
    
  Agent 4 (Validator):
    - Checks for missing key papers
    - Verifies citations are correct
    - Assesses completeness of review

Evaluation Metrics:
  - Coverage: # of relevant papers found vs. known total
  - Accuracy: % of correctly extracted findings (manual check sample)
  - Time: hours to complete vs. human baseline
  - Novelty: identifies gaps/future research directions

Hackathon MVP:
  - Use arXiv or PubMed API for paper retrieval
  - Parse abstracts with regex + LLM
  - Generate synthesis in ~2 min
  
Judges Will Love:
  - Systematic approach to literature review
  - Measurable quality metrics
  - LangGraph shows clear pipeline: Search → Extract → Synthesize → Validate
"""


# ============================================================================
# IDEA 2: HYPOTHESIS GENERATION & TESTING
# ============================================================================

PROJECT_2 = """
Name: Automated Hypothesis Generator for Protein Interactions
Domain: Molecular Biology / Biochemistry

Problem: Researchers manually formulate hypotheses based on gene/protein data.

Multi-Agent Solution:
  Agent 1 (Data Loader):
    - Loads protein interaction database (real or mock)
    - Extracts relevant features (structure, function, pathways)
    
  Agent 2 (Hypothesis Generator):
    - Analyzes data
    - Proposes testable hypotheses based on patterns
    - Ranks hypotheses by novelty & feasibility
    
  Agent 3 (Literature Validator):
    - Checks if hypothesis contradicts existing papers
    - Identifies supporting evidence
    - Flags potential issues
    
  Agent 4 (Experiment Designer):
    - Proposes wet-lab or computational experiments
    - Predicts likely outcomes
    - Estimates cost/time/difficulty

Evaluation Metrics:
  - Hypothesis novelty: not in literature (verified by search)
  - Testability: can the hypothesis be designed as an experiment?
  - Feasibility: cost/time estimate reasonable?
  - Biological plausibility: does it make sense?

Hackathon MVP:
  - Mock protein interaction data (JSON file)
  - Generate 5-10 hypotheses per dataset
  - Validate against Wikipedia/papers using web search
  
Judges Will Love:
  - Clear scientific workflow (observe → hypothesize → validate → design)
  - Novel use of agents for discovery
  - Metrics show how good the hypotheses are
"""


# ============================================================================
# IDEA 3: CLIMATE DATA ANALYSIS & FORECASTING
# ============================================================================

PROJECT_3 = """
Name: Multi-Agent Climate Impact Assessment
Domain: Environmental Science / Climate

Problem: Complex climate data is hard to interpret; scientists need actionable insights.

Multi-Agent Solution:
  Agent 1 (Data Preprocessor):
    - Loads climate datasets (temperature, CO2, sea level, etc.)
    - Cleans and normalizes data
    - Identifies missing values & anomalies
    
  Agent 2 (Trend Analyzer):
    - Analyzes historical trends
    - Performs statistical tests for significance
    - Generates key statistics (rate of change, etc.)
    
  Agent 3 (Risk Assessor):
    - Maps trends to impacts (flooding, crop failure, heat stress)
    - Quantifies risk in specific regions
    - Identifies tipping points
    
  Agent 4 (Mitigation Suggester):
    - Proposes policy/action recommendations
    - Estimates effectiveness based on literature
    - Prioritizes by impact/feasibility

Evaluation Metrics:
  - Data quality: % correctly processed
  - Trend accuracy: historical predictions vs. actual data
  - Risk calibration: assessed risk vs. realized outcomes (backtesting)
  - Recommendation quality: align with IPCC/scientific consensus

Hackathon MVP:
  - Use public climate dataset (e.g., NOAA, NASA)
  - Analyze regional data (e.g., UK temperature + rainfall last 20 years)
  - Generate risk report for a specific region
  
Judges Will Love:
  - Concrete, real-world problem
  - Multi-agent system handles complexity naturally
  - Tangible metrics on prediction accuracy
"""


# ============================================================================
# IDEA 4: DRUG DISCOVERY PIPELINE
# ============================================================================

PROJECT_4 = """
Name: AI-Assisted Drug Target Prioritization
Domain: Pharmaceutical Research / Structural Biology

Problem: Identifying promising drug targets is slow and expensive.

Multi-Agent Solution:
  Agent 1 (Disease Analyzer):
    - Takes disease name
    - Extracts known genes/proteins involved
    - Identifies validated targets in literature
    
  Agent 2 (Target Evaluator):
    - Scores targets on: druggability, specificity, safety
    - Checks for known inhibitors
    - Identifies structural features
    
  Agent 3 (Compound Finder):
    - Searches databases for known/hypothetical inhibitors
    - Predicts binding affinity (mock or real model)
    - Estimates ADMET properties
    
  Agent 4 (Risk Assessor):
    - Flags off-target effects
    - Predicts clinical trial failure modes
    - Assesses regulatory pathway difficulty

Evaluation Metrics:
  - Target accuracy: matches known validated targets?
  - Compound ranking: known drugs rank high?
  - Time: identification in < 2 min vs. human weeks
  - Feasibility: proposed targets have tractable chemistry?

Hackathon MVP:
  - Use disease gene associations database (mock or real)
  - Score targets on 3-4 simple criteria
  - Rank inhibitors by predicted affinity
  - Show one end-to-end example (e.g., disease → top target → lead compound)
  
Judges Will Love:
  - High-stakes scientific problem (drug discovery is $$$)
  - Multi-agent complexity mirrors real workflows
  - Concrete ranking/scoring methodology
"""


# ============================================================================
# IDEA 5: EXPERIMENTAL DESIGN & TROUBLESHOOTING
# ============================================================================

PROJECT_5 = """
Name: Intelligent Lab Protocol Assistant
Domain: Wet Lab Science (any field)

Problem: When experiments fail, troubleshooting is ad-hoc and time-consuming.

Multi-Agent Solution:
  Agent 1 (Protocol Analyzer):
    - Takes experimental protocol and results
    - Extracts key steps, conditions, measurements
    - Identifies critical parameters
    
  Agent 2 (Result Interpreter):
    - Analyzes outcome (success/failure)
    - Compares to expected results
    - Identifies anomalies or unexpected findings
    
  Agent 3 (Troubleshooter):
    - Hypothesizes failure causes
    - Ranks by likelihood
    - Proposes specific tests/modifications
    
  Agent 4 (Optimizers):
    - Suggests parameter changes to improve yield/accuracy
    - Estimates impact of changes
    - Ranks suggestions by impact/effort

Evaluation Metrics:
  - Diagnosis accuracy: troubleshooting suggests correct fix?
  - Optimization: proposed changes actually improve outcomes?
  - Time: actionable suggestions in < 5 min?
  - Feasibility: suggestions are practical for the lab?

Hackathon MVP:
  - Create 5 mock lab scenario (success + 4 failure modes)
  - For each, have agents diagnose root cause
  - Suggest specific fix or next experiment
  - Compare agent suggestions to "expert" human answers (peer from CS dept)
  
Judges Will Love:
  - Directly applicable to real labs (practical value)
  - Multi-agent system mirrors how lab teams think
  - Clear metrics on diagnostic accuracy
"""


# ============================================================================
# IDEA 6: SCIENTIFIC WRITING & GRANT PROPOSAL ANALYSIS
# ============================================================================

PROJECT_6 = """
Name: Automated Research Proposal Reviewer & Improver
Domain: Research Administration / Funding

Problem: Grant proposals are long; reviewers miss key issues; authors revise blindly.

Multi-Agent Solution:
  Agent 1 (Structure Checker):
    - Analyzes proposal format and completeness
    - Flags missing sections or low information density
    - Suggests reorganization for clarity
    
  Agent 2 (Technical Evaluator):
    - Assesses technical feasibility of proposed methods
    - Identifies risks and mitigation strategies
    - Checks for circular logic or unsupported claims
    
  Agent 3 (Significance Assessor):
    - Evaluates impact (novelty, scope, importance)
    - Checks against existing literature
    - Scores likely value to the field
    
  Agent 4 (Reviewer Simulator):
    - Predicts reviewer critiques based on proposal text
    - Generates typical reviewer comments
    - Suggests preemptive improvements

Evaluation Metrics:
  - Prediction accuracy: agent criticisms match real reviewers?
  - Improvement quality: authors act on suggestions → better scores?
  - Time: full review in < 10 min?
  - Coverage: catches common issues (significance, feasibility, etc.)?

Hackathon MVP:
  - Use sample grant proposals (anonymized, public examples online)
  - Generate structured review (strengths, weaknesses, feasibility score)
  - Compare agent review to actual funder feedback
  - Show before/after improvement suggestions
  
Judges Will Love:
  - High-value problem (grants = research funding)
  - Multi-agent system distributes expertise (structure, methods, impact, reviews)
  - Measurable accuracy (compare to real reviews)
"""


# ============================================================================
# QUICK SELECTION GUIDE
# ============================================================================

SELECTION_GUIDE = """
Choose based on:

1. DATA AVAILABILITY (crucial for hackathon)
   - Idea 1 (Literature): Easy—use arXiv/PubMed APIs
   - Idea 2 (Protein): Use mock data or UniProt
   - Idea 3 (Climate): Public datasets widely available
   - Idea 4 (Drug): Mock data or ChEMBL
   - Idea 5 (Lab): Mock protocols you write
   - Idea 6 (Grants): Use sample proposals online

2. COMPLEXITY
   - Easiest: Ideas 1, 5, 6 (mostly text/LLM)
   - Medium: Idea 2 (needs some biology knowledge)
   - Hardest: Idea 3, 4 (scientific domain depth)

3. JUDGES' LIKELY INTERESTS
   - Most impressive: Ideas 2, 4 (real scientific problems)
   - Most measurable: Ideas 3, 6 (concrete metrics)
   - Fastest to MVP: Ideas 1, 5 (quick to demo)

4. TEAM COMPOSITION
   - CS-heavy team? → Ideas 1, 3, 5 (engineering-focused)
   - Domain expertise (bio/chem)? → Ideas 2, 4
   - Mixed team? → Ideas 3, 6 (balanced)
"""


# ============================================================================
# MONDAY MORNING TEAM DISCUSSION TEMPLATE
# ============================================================================

DISCUSSION_TEMPLATE = """
When you find your team at 09:30, discuss:

1. What scientific problem excites us? (5 min)
   → Pick from these ideas or suggest your own

2. Do we have data access? (5 min)
   → Can we get real data? Mock data? Both?

3. Who will work on what? (5 min)
   → Agent design, integration, evaluation, presentation

4. What will success look like? (10 min)
   → 1-2 metrics we MUST hit
   → What would impress judges?

5. Quick prototype plan (5 min)
   → By 12:00 today, have 3 agents running end-to-end

Don't overthink—pick something fun and move fast.
The judges care about methodology, not perfection.
"""


if __name__ == "__main__":
    print("=" * 70)
    print("SCIENTIFIC RESEARCH PROJECT IDEAS FOR LANGGRAPH HACKATHON")
    print("=" * 70)
    
    ideas = [
        ("Literature Review Automation", PROJECT_1),
        ("Hypothesis Generation", PROJECT_2),
        ("Climate Data Analysis", PROJECT_3),
        ("Drug Discovery Pipeline", PROJECT_4),
        ("Lab Protocol Assistant", PROJECT_5),
        ("Grant Proposal Analysis", PROJECT_6),
    ]
    
    print("\nQuick overview:\n")
    for i, (name, _) in enumerate(ideas, 1):
        print(f"{i}. {name}")
    
    print("\n" + SELECTION_GUIDE)
    print("\n" + DISCUSSION_TEMPLATE)
    
    print("\n" + "=" * 70)
    print("Read through these Monday morning with your team!")
    print("=" * 70)

