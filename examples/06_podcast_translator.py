"""
Example 6: Podcast/Video Translator

Transcribe audio, translate it, and generate speech in another language.
This is the foundation for dubbing/localization workflows.
"""

from dotenv import load_dotenv

load_dotenv()
import json
import os

from langchain_camb import (
    CambTranscriptionTool,
    CambTranslationTool,
    CambTTSTool,
)

# Language codes
ENGLISH = 1
SPANISH = 2


def main():
    # Initialize tools
    transcriber = CambTranscriptionTool()
    translator = CambTranslationTool()
    tts = CambTTSTool()

    # Step 1: Transcribe the audio
    # (Replace with your audio URL or file path)
    audio_url = "https://example.com/podcast_clip.mp3"

    print("Step 1: Transcribing audio...")
    print("(Skipping - replace audio_url with a real URL to test)")

    # Simulated transcription result
    transcription = {
        "text": "Welcome to our podcast! Today we're discussing the future of AI.",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Welcome to our podcast!"},
            {"start": 2.5, "end": 6.0, "text": "Today we're discussing the future of AI."},
        ]
    }
    print(f"Transcription: {transcription['text']}\n")

    # Step 2: Translate each segment
    print("Step 2: Translating to Spanish...")
    translated_segments = []
    for segment in transcription["segments"]:
        translated = translator.invoke({
            "text": segment["text"],
            "source_language": ENGLISH,
            "target_language": SPANISH,
        })
        translated_segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "original": segment["text"],
            "translated": translated,
        })
        print(f"  '{segment['text']}' -> '{translated}'")

    # Step 3: Generate Spanish audio for each segment
    print("\nStep 3: Generating Spanish audio...")
    audio_files = []
    for i, segment in enumerate(translated_segments):
        audio_path = tts.invoke({
            "text": segment["translated"],
            "language": "es-es",
            "voice_id": 147320,
            "output_format": "file_path",
        })
        audio_files.append(audio_path)
        print(f"  Segment {i + 1}: {audio_path}")

    print("\nDone! The translated audio segments are ready.")
    print("You can combine them with video editing software for dubbing.")


if __name__ == "__main__":
    main()
