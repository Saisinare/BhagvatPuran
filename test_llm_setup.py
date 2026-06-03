#!/usr/bin/env python3
"""
Test LLM provider configuration for RAGita
"""
import sys
from pathlib import Path
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

print("=" * 60)
print("🧪 RAGita LLM Provider Test")
print("=" * 60)
print()

# Check environment variables
print("📋 Checking Configuration...")
print("-" * 60)

from dotenv import load_dotenv
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY", "")
groq_model = os.getenv("GROQ_MODEL", "")
use_groq = os.getenv("USE_GROQ", "true").lower() in ("1", "true", "yes")

openai_key = os.getenv("OPENAI_API_KEY", "")
openai_model = os.getenv("OPENAI_MODEL", "")

print(f"GROQ Enabled: {use_groq}")
print(f"GROQ API Key: {'✅ Set' if groq_key else '❌ Not Set'}")
print(f"GROQ Model: {groq_model if groq_model else '❌ Not Set'}")
print()
print(f"OpenAI API Key: {'✅ Set' if openai_key else '❌ Not Set'}")
print(f"OpenAI Model: {openai_model if openai_model else '❌ Not Set (will use default)'}")
print()

# Test generation
print("🚀 Testing Answer Generation...")
print("-" * 60)

try:
    from RAG.pipeline.generator import generate_answer
    
    test_passages = [
        {
            'chapter': 2,
            'verse': 20,
            'english': 'The soul is never born and never dies. It is unborn, eternal, permanent, and primeval. It is not killed when the body is killed.'
        },
        {
            'chapter': 2,
            'verse': 23,
            'english': 'Weapons cannot cut it, fire cannot burn it, water cannot wet it, and wind cannot dry it.'
        }
    ]
    
    print("Testing with query: 'What is the nature of the soul?'")
    print()
    
    result = generate_answer(
        query="What is the nature of the soul?",
        candidates=test_passages,
        persona="scholarly"
    )
    
    # Check if it's a placeholder
    if "LOCAL_MODEL_PLACEHOLDER" in result:
        print("❌ FAILED: Received placeholder response")
        print()
        print("This means all LLM providers failed.")
        print()
        print("🔧 Solutions:")
        print("1. Add OpenAI API key to .env file")
        print("2. Wait for GROQ rate limit to reset")
        print("3. Try different GROQ model")
        print()
        print("See LLM_PROVIDER_SETUP.md for detailed instructions")
        print()
        print("Placeholder preview:")
        print(result[:300] + "...")
        sys.exit(1)
    else:
        print("✅ SUCCESS: LLM is working!")
        print()
        print("Generated Answer Preview:")
        print("-" * 60)
        print(result[:400] + "..." if len(result) > 400 else result)
        print("-" * 60)
        print()
        print("🎉 Your RAGita system is ready to use!")
        print()
        print("Next steps:")
        print("1. Start backend: python backend/main.py")
        print("2. Start frontend: python -m http.server 8080 (in frontend folder)")
        print("3. Open http://localhost:8080")
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    print()
    print("Check backend logs for more details")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
