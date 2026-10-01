import os
import warnings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

load_dotenv()

# Silence non-critical UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    print("Hello from my langchain course!")

    # Dynamic information
    information = """
    Zohran Kwame Mamdani[c] (born October 18, 1991) is an American politician who has served since 2026 as the 112th mayor of New York City. A member of the Democratic Party and the Democratic Socialists of America, he represented the 36th district in the New York State Assembly from 2021 to 2025. Mamdani is New York City's first Muslim and first Asian American mayor.
    Born in Kampala, Uganda, to parents of Indian descent, Mamdani moved to New York City at seven years old and graduated from Bowdoin College in 2014 with a bachelor's degree in Africana studies. After graduation, he worked as a housing counselor and rapper, and entered New York City politics as a campaign manager for Khader El-Yateem and Ross Barkan. He was first elected to the New York State Assembly in 2020, defeating five-term incumbent Aravella Simotas in the Democratic primary. Representing Astoria and Long Island City, he was reelected without opposition in 2022 and 2024.
    In October 2024, Mamdani announced his candidacy for mayor of New York City in the 2025 mayoral election. A democratic socialist, Mamdani campaigned on a progressive, affordability-focused platform, supporting fare-free city buses, universal child care, city-owned grocery stores, a rent freeze on rent-stabilized units, additional affordable housing units, and a $30 minimum wage by 2030. He also expressed support for LGBTQ rights, comprehensive public safety reform, and tax increases on corporations and those earning above $1 million annually. He won the Democratic primary in June 2025, defeating former governor Andrew Cuomo in an upset, and was elected mayor in the November general election.
    """

    # Template
    summary_template = """
    given the information {information} about a person I want you to create:
        1. A short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # # LLM with explicit API key
    # llm = ChatGoogleGenerativeAI(temperature=0,
    #                              model="gemini-3.5-flash-lite",
    #                              api_key=os.getenv("GOOGLE_API_KEY"))

    # LLM without explicit API key
    llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-3.5-flash-lite")

    # Ollama integration (if needed)
    # llm = ChatOllama(temperature=0, model="gemma3:270m")

    # Chain
    chain = summary_prompt_template | llm | StrOutputParser()

    # Chain v2
    chain_v2 = summary_prompt_template | llm | {
        "clean_text": StrOutputParser(),
        "raw_response": RunnablePassthrough()
    }

    # Response
    # response = chain.invoke(input={"information": information})
    response = chain_v2.invoke(input={"information": information})


    # when using StrOutputParser, the response is a string directly. do not need the .content attribute
    # print(response.content)
    # print(response)

    # for chain v2
    print(response["clean_text"])
    print(response["raw_response"])


if __name__ == "__main__":
    main()
