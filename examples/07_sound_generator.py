"""
Example 7: AI Sound & Music Generator

Generate music, sound effects, and ambient sounds from text descriptions.
Great for game development, video production, and creative projects.
"""

from dotenv import load_dotenv

load_dotenv()
import os

from langchain_camb import CambTextToSoundTool


def main():
    sound_gen = CambTextToSoundTool()

    # Generate background music
    print("Generating background music...")
    music = sound_gen.invoke({
        "prompt": "Calm ambient music with soft piano and gentle strings, suitable for meditation",
        "duration": 30,
        "audio_type": "music",
        "output_format": "file_path",
    })
    print(f"Music saved to: {music}\n")

    # Generate sound effect
    print("Generating sound effect...")
    sfx = sound_gen.invoke({
        "prompt": "Futuristic sci-fi door opening with hydraulic hiss",
        "duration": 3,
        "audio_type": "sound",
        "output_format": "file_path",
    })
    print(f"Sound effect saved to: {sfx}\n")

    # Generate ambient soundscape
    print("Generating ambient soundscape...")
    ambient = sound_gen.invoke({
        "prompt": "Peaceful forest ambiance with birds chirping, wind through leaves, and a distant stream",
        "duration": 60,
        "audio_type": "sound",
        "output_format": "file_path",
    })
    print(f"Ambient sound saved to: {ambient}\n")

    # More creative examples
    examples = [
        ("Upbeat electronic dance music with driving bass", "music", 20),
        ("Thunderstorm with heavy rain and occasional thunder", "sound", 30),
        ("Retro 8-bit video game jump sound", "sound", 1),
        ("Epic orchestral fanfare for victory screen", "music", 10),
    ]

    for prompt, audio_type, duration in examples:
        print(f"Generating: {prompt[:50]}...")
        result = sound_gen.invoke({
            "prompt": prompt,
            "duration": duration,
            "audio_type": audio_type,
        })
        print(f"  Saved to: {result}")

    print("\nDone! All sounds generated.")


if __name__ == "__main__":
    main()
