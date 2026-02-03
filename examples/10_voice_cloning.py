"""
Example 10: Voice Cloning

Clone a voice from a short audio sample (2+ seconds) and use it for TTS.
"""

from dotenv import load_dotenv

load_dotenv()
import os

from langchain_camb import CambVoiceCloneTool, CambTTSTool


def main():
    voice_clone = CambVoiceCloneTool()
    tts = CambTTSTool()

    # Step 1: Clone a voice from an audio sample
    # You need a 2+ second audio file of the voice you want to clone
    sample_audio_path = "/path/to/your/voice_sample.wav"

    print("Step 1: Cloning voice from audio sample...")
    print(f"Audio file: {sample_audio_path}")

    # Note: Replace with actual file path to test
    # clone_result = voice_clone.invoke({
    #     "voice_name": "My Custom Voice",
    #     "audio_file_path": sample_audio_path,
    #     "gender": 2,  # 1=Male, 2=Female
    #     "description": "A warm, friendly voice for customer service",
    # })
    # print(f"Voice cloned! Result: {clone_result}")

    # Simulated result for demo
    print("(Skipping - replace sample_audio_path with actual file to test)")
    cloned_voice_id = 12345  # This would come from clone_result

    # Step 2: Use the cloned voice for TTS
    print("\nStep 2: Using cloned voice for TTS...")

    # With the cloned voice ID, you can generate speech
    # audio = tts.invoke({
    #     "text": "Hello! This is my cloned voice speaking.",
    #     "language": "en-us",
    #     "voice_id": cloned_voice_id,
    #     "output_format": "file_path",
    # })
    # print(f"Audio generated: {audio}")

    print("\nVoice Cloning Use Cases:")
    print("1. Personalized TTS for apps")
    print("2. Create brand-specific voices")
    print("3. Preserve voices for accessibility")
    print("4. Localize content with consistent voice")


if __name__ == "__main__":
    main()
