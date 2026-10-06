from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from tavily import TavilyClient

# tavily = TavilyClient()
# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query (str): The search query.

#     Returns:
#         str: The search results.
#     """
#     print(f"Results for: {query}")
#     return tavily.search(query=query)


llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello,from search agent?")
    # result = agent.ßinvoke(
    #     {"messages": HumanMessage(content="What is the weather in Tokyo?")}
    # )
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for 3 job postings for an ai engineer using langchain in the miami area on linkedin and list their details?"
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()
