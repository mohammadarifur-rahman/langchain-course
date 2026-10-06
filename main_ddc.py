from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:latest", temperature=0)
tools = [DuckDuckGoSearchResults(max_results=5, output_format="list")]
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a web search assistant. Use DuckDuckGo to find current information. "
        "Reply with only the final answer. Do not mention tools, reasoning steps, "
        "or internal messages."
    ),
)


def extract_answer(result) -> str:
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage) and message.content and not message.tool_calls:
            return message.content
    return "No answer generated."


def main():
    print("DuckDuckGo search chat (Ollama). Type 'quit' to exit.")
    messages = []

    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        messages.append(HumanMessage(content=query))
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        print(f"\nAssistant: {extract_answer(result)}")


if __name__ == "__main__":
    main()
