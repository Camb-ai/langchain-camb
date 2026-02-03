"""
Example 5: Multilingual Voice Assistant (using Gemini)

Build a voice assistant that can respond in any language.
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


def create_voice_assistant():
    """Create a multilingual voice assistant agent."""
    # Use only the tools we need for a voice assistant
    toolkit = CambToolkit(
        include_tts=True,
        include_translated_tts=True,
        include_translation=True,
        include_transcription=True,
        include_voice_list=True,
        include_voice_clone=False,  # Not needed for basic assistant
        include_text_to_sound=False,
        include_audio_separation=False,
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=1.0,  # Recommended for Gemini 3.0+
    )
    agent = create_react_agent(llm, toolkit.get_tools())

    return agent


def main():
    agent = create_voice_assistant()

    # Simulate user requests to the voice assistant
    requests = [
        "Say 'Hello, I am your AI assistant' in English",
        "Now say the same thing in Spanish",
        "Translate 'How can I help you today?' to French and speak it",
        "What languages can you speak in?",
    ]

    for request in requests:
        print(f"\n{'='*60}")
        print(f"User: {request}")
        print("=" * 60)

        result = agent.invoke({
            "messages": [{"role": "user", "content": request}]
        })

        print(f"Assistant: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
