
"""
Simple LLM Search Strategy Experiment
--------------------------------------

Goal:
1. Ask the LLM what needs to be done.
2. Run the same task using different search approaches.
3. Ask the LLM to compare the results.
4. Identify the best approach.
"""

# ============================================================
# 1. IMPORTS & SETUP
# ============================================================

import os
import logging
import warnings

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain.tools import tool

from tavily import TavilyClient


# Load environment variables
load_dotenv()

# Reduce noisy SDK logs
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.ERROR)
os.environ["GLOG_minloglevel"] = "2"


# ============================================================
# 2. CONFIGURATION
# ============================================================

MODEL = "gemini-3.5-flash-lite"

QUERY = """
Search for 3 AI Engineer job postings requiring LangChain
in the Miami area on LinkedIn.

For each job, provide:
- Job title
- Company
- Location
- Salary if available
- LangChain requirement
- URL
"""


# Create ONE shared LLM
llm = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0,
)


# Native Tavily client
tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# ============================================================
# 3. STEP 1 — ASK LLM TO PLAN
# ============================================================

def create_plan(query):
    """
    Ask the LLM to determine what needs to be done
    before searching.
    """

    prompt = f"""
You are planning a web research task.

User request:
{query}

Before doing the search, explain briefly:

1. What information needs to be found?
2. What should the search focus on?
3. How should the results be judged?

Do not answer the user's question yet.
Just create a short search plan.
"""

    response = llm.invoke(prompt)

    return response.content


# ============================================================
# 4. STEP 2 — SEARCH APPROACHES
# ============================================================

def run_agent(tool, approach_name, query, plan):
    """
    Run one search approach.

    If a search provider fails, return the error instead
    of stopping the entire experiment.
    """

    print(f"\n{'=' * 60}")
    print(f"Running: {approach_name}")
    print(f"{'=' * 60}")

    try:
        agent = create_agent(
            model=llm,
            tools=[tool],
        )

        prompt = f"""
Complete this task:

{query}

The planning LLM created this plan:

{plan}

Use the available search tool to complete the task.

Return the best results you can find.
"""

        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        })

        answer = result["messages"][-1].content

        print(answer)

        return answer

    except Exception as e:
        error = f"{type(e).__name__}: {e}"

        print(f"❌ {approach_name} failed:")
        print(error)

        return f"SEARCH FAILED: {error}"


# ============================================================
# 5. CUSTOM TAVILY TOOL
# ============================================================

@tool
def custom_tavily_search(query: str):
    """
    Search the web using the native Tavily SDK.
    """

    response = tavily_client.search(
        query=query,
        max_results=5,
    )

    return response.get("results", [])


# ============================================================
# 6. STEP 3 — COMPARE RESULTS
# ============================================================

def compare_results(query, plan, results):
    """
    Ask the LLM to compare the different approaches.
    """

    comparison = f"""
You are evaluating different web-search approaches.

Original task:
{query}

Search plan:
{plan}

Here are the results from each approach:

--------------------------------------------------
MODERN TAVILY
--------------------------------------------------
{results["Modern Tavily"]}

--------------------------------------------------
LEGACY TAVILY
--------------------------------------------------
{results["Legacy Tavily"]}

--------------------------------------------------
DUCKDUCKGO
--------------------------------------------------
{results["DuckDuckGo"]}

--------------------------------------------------
CUSTOM TAVILY
--------------------------------------------------
{results["Custom Tavily"]}


Compare the four approaches.

Evaluate:

- Relevance
- Accuracy
- Completeness
- Source quality
- How well the task was followed

Give each approach a score from 1-10.

Then identify:

1. The best approach
2. Why it performed best
3. The biggest weakness of each approach
"""

    response = llm.invoke(comparison)

    return response.content


# ============================================================
# 7. MAIN EXPERIMENT
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("LLM SEARCH STRATEGY EXPERIMENT")
    print("=" * 60)

    # --------------------------------------------------------
    # Step 1: Planning
    # --------------------------------------------------------

    print("\nSTEP 1 — LLM PLANNING\n")

    plan = create_plan(QUERY)

    print(plan)


    # --------------------------------------------------------
    # Step 2: Run different approaches
    # --------------------------------------------------------

    print("\nSTEP 2 — SEARCH EXPERIMENTS")

    results = {}

    results["Modern Tavily"] = run_agent(
        TavilySearch(max_results=5),
        "Modern Tavily",
        QUERY,
        plan,
    )

    results["Legacy Tavily"] = run_agent(
        TavilySearchResults(max_results=5),
        "Legacy Tavily",
        QUERY,
        plan,
    )

    results["DuckDuckGo"] = run_agent(
        DuckDuckGoSearchRun(),
        "DuckDuckGo",
        QUERY,
        plan,
    )

    results["Custom Tavily"] = run_agent(
        custom_tavily_search,
        "Custom Tavily",
        QUERY,
        plan,
    )


    # --------------------------------------------------------
    # Step 3: Let LLM compare everything
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("STEP 3 — LLM COMPARISON")
    print("=" * 60)

    evaluation = compare_results(
        QUERY,
        plan,
        results,
    )

    print(evaluation)


# ============================================================
# 8. RUN
# ============================================================

if __name__ == "__main__":
    main()
