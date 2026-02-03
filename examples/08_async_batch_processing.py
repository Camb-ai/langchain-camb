"""
Example 8: Async Batch Processing

Process multiple audio tasks concurrently for better performance.
"""

from dotenv import load_dotenv

load_dotenv()
import asyncio
import os
import time

from langchain_camb import CambTTSTool, CambTranslationTool


async def main():
    tts = CambTTSTool()
    translator = CambTranslationTool()

    # Text to process in multiple languages
    messages = [
        "Welcome to our service!",
        "Thank you for your purchase.",
        "Have a great day!",
    ]

    # Language configurations (code, lang_id, name)
    languages = [
        ("es-es", 54, "Spanish"),   # es-es
        ("fr-fr", 76, "French"),    # fr-fr
        ("de-de", 31, "German"),    # de-de
        ("ja-jp", 88, "Japanese"),  # ja-jp
    ]

    print("Processing messages in multiple languages concurrently...\n")
    start_time = time.time()

    # Create all tasks
    tasks = []
    for message in messages:
        for lang_code, lang_id, lang_name in languages:
            # Create translation task
            async def process(msg, lc, li, ln):
                # Translate
                translated = await translator.ainvoke({
                    "text": msg,
                    "source_language": 1,  # English
                    "target_language": li,
                })

                # Generate TTS
                audio = await tts.ainvoke({
                    "text": translated,
                    "language": lc,
                    "voice_id": 147320,
                    "output_format": "file_path",
                })

                return {
                    "original": msg,
                    "language": ln,
                    "translated": translated,
                    "audio": audio,
                }

            tasks.append(process(message, lang_code, lang_id, lang_name))

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    print(f"Processed {len(results)} audio files in {elapsed:.2f} seconds\n")

    # Display results
    for result in results:
        print(f"[{result['language']}] {result['original']}")
        print(f"  -> {result['translated']}")
        print(f"  -> Audio: {result['audio']}\n")


if __name__ == "__main__":
    asyncio.run(main())
