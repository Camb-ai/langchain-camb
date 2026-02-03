"""
Example 3: Translated Text-to-Speech

Translate text AND generate speech in the target language - all in one step.
Perfect for building multilingual voice assistants.
"""

from dotenv import load_dotenv

load_dotenv()

from langchain_camb import CambTranslatedTTSTool

# Language codes (use client.languages.get_source_languages() to see all)
ENGLISH = 1      # en-us
SPANISH = 54     # es-es (Spain)
FRENCH = 76      # fr-fr (France)
GERMAN = 31      # de-de (Germany)
JAPANESE = 88    # ja-jp


def main():
    translated_tts = CambTranslatedTTSTool()

    # Translate and speak: English -> Spanish
    print("Translating and speaking in Spanish...")
    spanish_audio = translated_tts.invoke({
        "text": "Hello! Thank you for using our service. How can I help you today?",
        "source_language": ENGLISH,
        "target_language": SPANISH,
        "voice_id": 147320,
    })
    print(f"Spanish audio saved to: {spanish_audio}\n")

    # Translate and speak: English -> French (formal)
    print("Translating and speaking in French (formal)...")
    french_audio = translated_tts.invoke({
        "text": "We appreciate your business. Please let us know if you need assistance.",
        "source_language": ENGLISH,
        "target_language": FRENCH,
        "voice_id": 147320,
        "formality": 1,  # 1=formal, 2=informal
    })
    print(f"French audio saved to: {french_audio}\n")

    # Translate and speak: English -> Japanese
    print("Translating and speaking in Japanese...")
    japanese_audio = translated_tts.invoke({
        "text": "Welcome to our store. We hope you find what you're looking for.",
        "source_language": ENGLISH,
        "target_language": JAPANESE,
        "voice_id": 147320,
    })
    print(f"Japanese audio saved to: {japanese_audio}\n")

    print("Done! These audio files contain the translated speech.")


if __name__ == "__main__":
    main()
