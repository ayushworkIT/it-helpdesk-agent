"""
Step 1.10 - First LangChain program: a simple IT help desk chain.
Flow: prompt template -> LLM -> plain-text answer
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load OPENAI_API_KEY and OPENAI_MODEL from .env
load_dotenv()

# 2. The model. os.getenv reads OPENAI_MODEL; "gpt-5.4-mini" is a fallback
#    in case the line is missing from .env
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))

# 3. The prompt template: system = role/rules, human = the user's question
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a friendly IT help desk assistant at RakFort. "
     "Give short, clear, numbered troubleshooting steps that "
     "non-technical staff can follow."),
    ("human", "{question}"),
])

# 4. Build the chain with the pipe operator: prompt -> model -> text
chain = prompt | llm | StrOutputParser()

# 5. Run it once with a realistic help desk question
answer = chain.invoke(
    {"question": "My laptop can't connect to the office Wi-Fi. What should I try first?"}
)
print(answer)