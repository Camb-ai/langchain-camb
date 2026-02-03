"""
Example 9: Multilingual Customer Support Bot (using Gemini)

A complete example of a customer support chatbot that can:
- Respond in the customer's preferred language
- Generate audio responses
- Handle multiple languages seamlessly
"""

from dotenv import load_dotenv

load_dotenv()
import os
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from langchain_camb import CambToolkit

# Set your API keys
# os.environ["CAMB_API_KEY"] = "your-camb-api-key"
# os.environ["GOOGLE_API_KEY"] = "your-google-api-key"


SYSTEM_PROMPT = """You are a helpful multilingual customer support agent for TechCorp.

Your capabilities:
1. You can respond in ANY language the customer uses
2. You can generate audio responses using text-to-speech
3. You can translate between languages

Guidelines:
- Detect the customer's language and respond in the same language
- When asked to speak, use the camb_tts tool to generate audio
- For translated audio, use camb_translated_tts
- Be helpful, friendly, and professional
- Keep responses concise

Available products: TechPhone Pro ($999), TechWatch ($299), TechBuds ($149)
Support hours: 24/7
Return policy: 30 days, no questions asked
"""


def create_support_bot():
    """Create the customer support bot."""
    toolkit = CambToolkit(
        include_tts=True,
        include_translated_tts=True,
        include_translation=True,
        include_voice_list=True,
        # Disable tools we don't need
        include_transcription=False,
        include_voice_clone=False,
        include_text_to_sound=False,
        include_audio_separation=False,
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=1.0,  # Recommended for Gemini 3.0+
    )
    agent = create_react_agent(llm, toolkit.get_tools())

    return agent


def chat(agent, message: str, generate_audio: bool = False):
    """Send a message to the support bot."""
    if generate_audio:
        message += " Please also generate an audio response."

    result = agent.invoke({
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=message),
        ]
    })

    return result["messages"][-1].content


def main():
    print("=" * 60)
    print("TechCorp Multilingual Customer Support (Powered by Gemini)")
    print("=" * 60)
    print()

    agent = create_support_bot()

    # Simulate customer interactions in different languages

    # English customer
    print("Customer (English): What's your return policy?")
    response = chat(agent, "What's your return policy?")
    print(f"Bot: {response}\n")

    # Spanish customer
    print("Customer (Spanish): ¿Cuánto cuesta el TechPhone Pro?")
    response = chat(agent, "¿Cuánto cuesta el TechPhone Pro?")
    print(f"Bot: {response}\n")

    # French customer wants audio
    print("Customer (French): Bonjour, pouvez-vous me dire vos heures d'ouverture?")
    response = chat(agent, "Bonjour, pouvez-vous me dire vos heures d'ouverture?", generate_audio=True)
    print(f"Bot: {response}\n")

    # Japanese customer
    print("Customer (Japanese): TechBudsの価格を教えてください")
    response = chat(agent, "TechBudsの価格を教えてください")
    print(f"Bot: {response}\n")

    # German customer
    print("Customer (German): Ich möchte meine TechWatch zurückgeben. Wie geht das?")
    response = chat(agent, "Ich möchte meine TechWatch zurückgeben. Wie geht das?")
    print(f"Bot: {response}\n")

    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
