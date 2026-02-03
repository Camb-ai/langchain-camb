"""
Quick Test - Run this to verify your setup works!

This script tests the basic CAMB AI tools without requiring OpenAI API key.
"""

from dotenv import load_dotenv

load_dotenv()
import os
import sys

# Check for API key
if not os.environ.get("CAMB_API_KEY"):
    print("Error: CAMB_API_KEY environment variable not set!")
    print("\nTo set it:")
    print("  export CAMB_API_KEY='your-api-key'")
    print("\nGet your API key at: https://camb.ai")
    sys.exit(1)

from langchain_camb import (
    CambToolkit,
    CambTTSTool,
    CambTranslationTool,
    CambVoiceListTool,
)


def test_voice_list():
    """Test listing voices."""
    print("\n1. Testing Voice List...")
    try:
        tool = CambVoiceListTool()
        result = tool.invoke({})
        print(f"   ✓ Found voices! First 200 chars: {result[:200]}...")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def test_translation():
    """Test translation."""
    print("\n2. Testing Translation...")
    try:
        tool = CambTranslationTool()
        result = tool.invoke({
            "text": "Hello, world!",
            "source_language": 1,   # English (en-us)
            "target_language": 54,  # Spanish (es-es)
        })
        print(f"   ✓ Translated 'Hello, world!' to: {result}")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def test_tts():
    """Test text-to-speech."""
    print("\n3. Testing Text-to-Speech...")
    try:
        tool = CambTTSTool()
        result = tool.invoke({
            "text": "Testing CAMB AI text to speech.",
            "language": "en-us",
            "voice_id": 147320,
            "output_format": "file_path",
        })
        print(f"   ✓ Generated audio: {result}")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def test_toolkit():
    """Test toolkit."""
    print("\n4. Testing Toolkit...")
    try:
        toolkit = CambToolkit()
        tools = toolkit.get_tools()
        print(f"   ✓ Toolkit has {len(tools)} tools: {[t.name for t in tools]}")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def main():
    print("=" * 50)
    print("langchain-camb Quick Test")
    print("=" * 50)

    results = []
    results.append(("Voice List", test_voice_list()))
    results.append(("Translation", test_translation()))
    results.append(("TTS", test_tts()))
    results.append(("Toolkit", test_toolkit()))

    print("\n" + "=" * 50)
    print("Results:")
    print("=" * 50)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Your setup is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check your API key and network connection.")


if __name__ == "__main__":
    main()
