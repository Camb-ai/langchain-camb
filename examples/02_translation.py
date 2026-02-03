"""
Example 2: Text Translation

Translate text between 140+ languages.
"""

from dotenv import load_dotenv

load_dotenv()
import os

from langchain_camb import CambTranslationTool

# Language codes (use client.languages.get_source_languages() to see all)
LANGUAGES = {
    "english": 1,      # en-us
    "spanish": 54,     # es-es (Spain)
    "french": 76,      # fr-fr (France)
    "german": 31,      # de-de (Germany)
    "italian": 87,     # it-it
    "portuguese": 111, # pt-br (Brazil)
    "dutch": 108,      # nl-nl
    "russian": 114,    # ru-ru
    "japanese": 88,    # ja-jp
    "korean": 94,      # ko-kr
    "chinese": 139,    # zh-cn (Simplified)
}


def main():
    translator = CambTranslationTool()

    # Simple translation
    print("Translating 'Hello, how are you?' to Spanish...")
    spanish = translator.invoke({
        "text": "Hello, how are you?",
        "source_language": LANGUAGES["english"],
        "target_language": LANGUAGES["spanish"],
    })
    print(f"Spanish: {spanish}\n")

    # Formal translation
    print("Translating with formal tone to German...")
    german_formal = translator.invoke({
        "text": "Can you help me with this problem?",
        "source_language": LANGUAGES["english"],
        "target_language": LANGUAGES["german"],
        "formality": 1,  # 1=formal, 2=informal
    })
    print(f"German (formal): {german_formal}\n")

    # Informal translation
    print("Translating with informal tone to French...")
    french_informal = translator.invoke({
        "text": "What's up? Want to hang out later?",
        "source_language": LANGUAGES["english"],
        "target_language": LANGUAGES["french"],
        "formality": 2,  # 1=formal, 2=informal
    })
    print(f"French (informal): {french_informal}\n")

    # Multi-language translation
    print("Translating 'Good morning' to multiple languages...")
    text = "Good morning! Have a wonderful day."

    for lang_name, lang_code in [("spanish", 54), ("french", 76), ("japanese", 88)]:
        result = translator.invoke({
            "text": text,
            "source_language": LANGUAGES["english"],
            "target_language": lang_code,
        })
        print(f"  {lang_name.capitalize()}: {result}")


if __name__ == "__main__":
    main()
