import os
from dotenv import load_dotenv
from langchain_classic.agents import tool, create_tool_calling_agent, AgentExecutor
import datetime

load_dotenv()

@tool
def get_current_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Returns the current date and time, formatted according to the provided Python strftime format string.
    Use this tool whenever the user asks for the current date, time, or both.
    Example format strings: '%Y-%m-%d' for date, '%H:%M:%S' for time.
    If no format is specified, defaults to '%Y-%m-%d %H:%M:%S'.
    """
    try:
        return datetime.datetime.now().strftime(format)
    except Exception as e:
        return f"Error formatting date/time: {e}"
    
tools = [get_current_datetime]
print("custom tool defined")

from langchain_ollama import ChatOllama

def get_agent_llm(model_name="quen3:8b", temperature=0):
    """Initializes the ChatOllama model for the agent."""
    # we have to ensure teh ollama server is running in the backgrounmd for this to work
    llm = ChatOllama(
        model = model_name,
        temperature=temperature
    )

    print(f"Initialized teh ChatOllama agent LLM with model: {model_name}")
    return llm

# create the agent prompt now

try:
    # Newer LangChain versions may provide a hub API to pull prompt templates.
    from langchain import hub


    def get_agent_prompt(prompt_hub_name: str = "hwchase17/openai-tools-agent"):
        """Pulls the agent prompt template from LangChain Hub when available."""
        prompt = hub.pull(prompt_hub_name)
        print(f"Pulled agent prompt from Hub: {prompt_hub_name}")
        return prompt

except Exception:
    # Fallback: construct a minimal ChatPromptTemplate compatible with
    # create_tool_calling_agent. The prompt must include an `agent_scratchpad`
    # MessagesPlaceholder variable as used by the tool-calling pipeline.
    from langchain_core.prompts.chat import ChatPromptTemplate


    def get_agent_prompt(prompt_hub_name: str = "hwchase17/openai-tools-agent"):
        """Fallback prompt builder when `langchain.hub` is not present."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant that can call tools."),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        print("Using local fallback ChatPromptTemplate (langchain.hub not available)")
        return prompt

def build_agent(llm, tools, prompt):
    """Builds teh tool calling agent runnable"""
    agent = create_tool_calling_agent(llm, tools, prompt)
    print("Agent runnable created")
    return agent

# Note: using `langchain_classic.agents` because the installed `langchain` package
# delegates older agent APIs to `langchain_classic`. The `tool` decorator,
# `create_tool_calling_agent` and `AgentExecutor` are provided there.
def create_agent_executor(agent, tools):
    """create teh agent executor"""
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )
    print('Agent executor created')
    return agent_executor

def run_agent(executor, user_input):
    """Runs teh agent executor with the given input"""
    print("\nInvoking agent. . .")
    print(f"Input: {user_input}")
    response = executor.invoke({"input":user_input})
    print("\nAgent Response:")
    print(response['output'])

if __name__ == "__main__":
    agent_llm = get_agent_llm(model_name="qwen3:8b")
    agent_prompt = get_agent_prompt()
    # pass the list of tool functions (tools) to the builder
    agent_runnable = build_agent(agent_llm, tools, agent_prompt)
    agent_executor = create_agent_executor(agent_runnable, tools)

    run_agent(agent_executor, "What is the current date?")
    run_agent(agent_executor, "What time is it right now? Use HH:MM format.")
    run_agent(agent_executor, "Tell me a good joke about the current time")