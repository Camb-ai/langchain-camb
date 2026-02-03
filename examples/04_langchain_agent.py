"""
Example 4: LangChain Agent with CAMB AI Tools (using Gemini)

Create an AI agent that can speak, translate, and process audio.
This is the most powerful way to use CAMB AI with LangChain!
"""

from dotenv import load_dotenv

load_dotenv()
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from langchain_camb import CambToolkit

# Set your API keys
# os.environ["CAMB_API_KEY"] = "your-camb-api-key"
# os.environ["GOOGLE_API_KEY"] = "your-google-api-key"


def main():
    # Create the toolkit with all CAMB AI tools
    toolkit = CambToolkit()
    tools = toolkit.get_tools()

    print(f"Available tools: {[t.name for t in tools]}\n")

    # Create the agent with Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=1.0,  # Recommended for Gemini 3.0+
    )
    agent = create_react_agent(llm, tools)

    # Example 1: Generate speech
    print("=" * 50)
    print("Request: Say 'Hello world' in English")
    print("=" * 50)
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Say 'Hello world' in English using text-to-speech"}]
    })
    print(f"Agent response: {result['messages'][-1].content}\n")

    # Example 2: Translate text
    print("=" * 50)
    print("Request: Translate a phrase")
    print("=" * 50)
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Translate 'I love programming' to Spanish and French"}]
    })
    print(f"Agent response: {result['messages'][-1].content}\n")

    # Example 3: Translate AND speak
    print("=" * 50)
    print("Request: Translate and speak")
    print("=" * 50)
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Translate 'Good morning, have a great day!' to Japanese and generate audio of it"}]
    })
    print(f"Agent response: {result['messages'][-1].content}\n")

    # Example 4: List voices
    print("=" * 50)
    print("Request: Find available voices")
    print("=" * 50)
    result = agent.invoke({
        "messages": [{"role": "user", "content": "What voices are available? Show me a few options."}]
    })
    print(f"Agent response: {result['messages'][-1].content}\n")

    # Example 5: Complex multi-step task
    print("=" * 50)
    print("Request: Complex task")
    print("=" * 50)
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": """
            I need to create a multilingual greeting for my app:
            1. First, find a good voice to use
            2. Then translate "Welcome to our app!" to Spanish
            3. Generate audio of that Spanish greeting
            """
        }]
    })
    print(f"Agent response: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
