import os
import logging
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from agents.tools import (
    search_articles,
    get_sentiment_summary,
    get_top_articles,
    get_trending_topics
)
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tools = [
    search_articles,
    get_sentiment_summary,
    get_top_articles,
    get_trending_topics
]

prompt = PromptTemplate(
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
    template="""You are a market intelligence analyst. Answer the question using the tools available.

TOOLS:
{tools}

FORMAT:

Question: the input question
Thought: what tool to use and why
Action: tool name from [{tool_names}]
Action Input: the input string only
Observation: tool result
Thought: I now know the final answer
Final Answer: your answer here

CRITICAL RULES:
- After ANY Observation that contains data, your very next line MUST be "Thought: I now know the final answer" followed by "Final Answer:"
- NEVER call a tool twice with the same input
- NEVER repeat a Question/Thought/Action block you already did
- If a tool returns no data, try once with different input, then write Final Answer with what you have
- Action Input must be a plain string only
- If search_articles returns unavailable, immediately use get_sentiment_summary or get_trending_topics instead
- Never retry a tool that returned an error

Question: {input}
{agent_scratchpad}"""
)


def get_agent_executor():
    """Build agent executor — call this once and cache it."""
    logger.info("Initializing LLM and agent...")

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=800
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=3,
        handle_parsing_errors="Check your format and continue."
    )


def run_agent(query: str, executor: AgentExecutor = None) -> str:
    """Run the agent. Accepts optional cached executor from Streamlit."""
    if executor is None:
        executor = get_agent_executor()

    logger.info(f"Running agent query: {query}")
    result = executor.invoke({"input": query})
    return result["output"]


if __name__ == "__main__":
    executor = get_agent_executor()
    queries = [
        "Which category has the most negative sentiment right now?",
        "Show me the top 3 most negative articles about stock market",
        "What topics are trending today and what's the overall mood?"
    ]
    for query in queries:
        answer = run_agent(query, executor)
        print(f"\nQ: {query}\nA: {answer}\n")