# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # LangChain
# Playing around

# %%

import os
import sys

# set PYTHONPATH env variable
sys.path.append("/Users/akusok/.pyenv/shims/python")

print(sys.executable)

# %%
from langchain_ollama import OllamaLLM, ChatOllama, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools, AgentExecutor, create_react_agent
from langchain import ConversationChain

import math
import numexpr

# %%

llm = OllamaLLM(model="phi3", temperature=0.9)

# %%

text = "What are 5 vacation destinations for someone who loves to eat pasta?"

print(llm.invoke(text))

# %% [markdown]
# # Prompt
# Creating standard queries with prompts

# %%

prompt = PromptTemplate(
    input_variables = ["food"],
    template = "What are 5 vacation destinations for someone who loves to eat {food}?",
)

print(prompt.format(food="dessert"))

# %%

print(llm.invoke(prompt.format(food="dessert")))

# %% [markdown]
# # Chain
# Multi-step workflows with chains

# %%

chain = prompt | llm

# %%

print(
    chain.invoke(input={"food": "dessert"})
)

# %% [markdown]
# # Agents that can do other things

# %%

@tool
def calculator(expression: str) -> str:
    """Calculator that runs mathetical expressions in `numexpr` and returns their result.

    Expression should be a single line mathematical expression
    that solves the problem.

    Examples:
        "37593 * 67" for "37593 times 67"
        "37593**(1/5)" for "37593^(1/5)"
    """
    local_dict = {"pi": math.pi, "e": math.e}
    return str(
        numexpr.evaluate(
            expression.strip(),
            global_dict={},  # restrict access to globals
            local_dict=local_dict,  # add common mathematical functions
        )
    )

tools = [calculator]

chat_model = ChatOllama(model="llama3.2")
llm_with_tools = chat_model.bind_tools(tools)

result = llm_with_tools.invoke("What is two plus three? Also what is 45 / 5?")

print(result.content)
print(result.tool_calls)

# %%

query = "What is 3 * 12? Also, what is 112367 + 26449 / 2462?"

messages = [HumanMessage(query)]

# %%

result = llm_with_tools.invoke(messages)
messages.append(result)

print(result.tool_calls)

# %%
# Generating tool messages from calls

for tool_call in result.tool_calls:
    tool_msg = calculator.invoke(tool_call)
    messages.append(tool_msg)

# %%
# Re-invoke the model with the tool messages

print(llm_with_tools.invoke(messages).content)









# %% [markdown]
# # Memory

# %%

conversation = ConversationChain(llm=chat_model, verbose=True)

# %%

print(
    conversation.predict(input="Hi there")
)

# %%

print(
    conversation.predict(input="I am doing well, just having a conversation with an AI.")
)

# %%

print(
    conversation.predict(input="Can you remind me the first thing I said to you?")
)

# %%

print(
    conversation.predict(input="What is an alternative phrasing for the second thing I said to you?")
)







# %% [markdown]
# # Nice tool template example, even if we don't use it

# %%

template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)
agent = create_react_agent(llm=llm_with_tools, tools=tools, prompt=prompt)

# %%

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    return_intermediate_steps=True,
    handle_parsing_errors=True,
)

agent_scratchpad = ""

agent_executor.invoke(
    input={"input": "What is 36562 plus 4727 divided by 72?", "agent_scratchpad": agent_scratchpad},
)




# %%

llm.invoke("The first man on the moon was ...")

# %%

for chunk in llm.stream("The first man on the moon was ..."):
    print(chunk, end="", flush=True)

# %%

chat_model = ChatOllama(model="llama3.2")

chat_model.invoke("Who was the first man on the moon?")

# %%

# Create a vector store with a sample text
text = "LangChain is the framework for building context-aware reasoning applications"

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

vectorstore = InMemoryVectorStore.from_texts(
    [text],
    embedding=embeddings,
)

# %%

# Use the vectorstore as a retriever
retriever = vectorstore.as_retriever()

# Retrieve the most similar text
retrieved_documents = retriever.invoke("What is LangChain?")

# show the retrieved document's content
retrieved_documents[0]

# %%
