from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from tools import tools

load_dotenv()

llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)

agent = create_agent(
    tools=tools,
    model=llm,
    system_prompt=(
        "You are a helpful shopping assistant."
    ),
)

if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I want to buy organic honey with 4.5+ rating and less than $20 price."
                    ),
                }
            ]
        }
    )
    print(result["messages"][-1].content)
