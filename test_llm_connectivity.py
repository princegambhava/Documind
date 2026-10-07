"""
Test script to verify LLM connectivity for Gemini and Groq.
Run this to test actual API calls without exposing credentials.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import (
    LLM_BACKEND, GOOGLE_API_KEY, GROQ_API_KEY,
    GEMINI_MODEL, GROQ_MODEL
)
from generator import LLMGenerator

def test_gemini():
    """Test Gemini API connectivity."""
    print("=" * 60)
    print("TESTING GEMINI")
    print("=" * 60)
    
    if not GOOGLE_API_KEY:
        print("❌ GOOGLE_API_KEY not configured")
        return False
    
    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=GOOGLE_API_KEY)
        
        print(f"Model: {GEMINI_MODEL}")
        print("Sending test request...")
        
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents="What is 2+2? Answer with just the number.",
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=100
            )
        )
        
        print(f"✅ Gemini response: {response.text.strip()}")
        return True
        
    except Exception as e:
        print(f"❌ Gemini error: {str(e)}")
        return False

def test_groq():
    """Test Groq API connectivity."""
    print("\n" + "=" * 60)
    print("TESTING GROQ")
    print("=" * 60)
    
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not configured")
        return False
    
    try:
        from groq import Groq
        
        client = Groq(api_key=GROQ_API_KEY)
        
        print(f"Model: {GROQ_MODEL}")
        print("Sending test request...")
        
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "user", "content": "What is 2+2? Answer with just the number."}
            ]
        )
        
        answer = response.choices[0].message.content
        print(f"✅ Groq response: {answer.strip()}")
        return True
        
    except Exception as e:
        print(f"❌ Groq error: {str(e)}")
        return False

def test_llm_generator():
    """Test LLMGenerator class with fallback."""
    print("\n" + "=" * 60)
    print("TESTING LLM GENERATOR WITH FALLBACK")
    print("=" * 60)
    
    try:
        generator = LLMGenerator()
        
        # Test with a simple question
        context = "Test context for testing purposes."
        question = "What is 2+2?"
        
        print(f"Selected backend: {generator.backend}")
        print("Sending test request through LLMGenerator...")
        
        result = generator.generate_answer(context, question)
        
        print(f"✅ Backend used: {result['backend']}")
        print(f"✅ Answer: {result['answer']}")
        return True
        
    except Exception as e:
        print(f"❌ LLMGenerator error: {str(e)}")
        return False

if __name__ == "__main__":
    print("LLM CONNECTIVITY TEST")
    print("=" * 60)
    
    gemini_ok = test_gemini()
    groq_ok = test_groq()
    generator_ok = test_llm_generator()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Gemini: {'✅ PASS' if gemini_ok else '❌ FAIL'}")
    print(f"Groq: {'✅ PASS' if groq_ok else '❌ FAIL'}")
    print(f"LLMGenerator: {'✅ PASS' if generator_ok else '❌ FAIL'}")
    print("=" * 60)
