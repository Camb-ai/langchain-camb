"""
Example 1: Basic Text-to-Speech

Generate speech from text in multiple languages.
"""

from dotenv import load_dotenv

load_dotenv()
import os

from langchain_camb import CambTTSTool, CambVoiceListTool

# Set your API key
# os.environ["CAMB_API_KEY"] = "your-api-key"


def main():
    # First, list available voices
    print("Fetching available voices...")
    voice_list = CambVoiceListTool()
    voices = voice_list.invoke({})
    print(f"Available voices (first 5):\n{voices[:500]}...\n")

    # Create TTS tool
    tts = CambTTSTool()

    # Generate speech in English
    print("Generating English speech...")
    english_audio = tts.invoke({
        "text": "Hello! Welcome to CAMB AI. We support over 140 languages for text to speech.",
        "language": "en-us",
        "voice_id": 147320,  # Default voice
        "speech_model": "mars-flash",
        "output_format": "file_path",
    })
    print(f"English audio saved to: {english_audio}")

    # Generate speech in Spanish
    print("\nGenerating Spanish speech...")
    spanish_audio = tts.invoke({
        "text": "¡Hola! Bienvenido a CAMB AI. Soportamos más de 140 idiomas.",
        "language": "es-es",
        "voice_id": 147320,
        "output_format": "file_path",
    })
    print(f"Spanish audio saved to: {spanish_audio}")

    # Generate with different speed
    print("\nGenerating slow speech...")
    slow_audio = tts.invoke({
        "text": "This is spoken slowly for clarity.",
        "language": "en-us",
        "voice_id": 147320,
        "speed": 0.7,  # Slower
        "output_format": "file_path",
    })
    print(f"Slow audio saved to: {slow_audio}")

    print("\nDone! You can play these audio files with any media player.")


if __name__ == "__main__":
    main()
