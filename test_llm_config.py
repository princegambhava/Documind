"""
Test script to verify LLM configuration and connectivity.
Run this to diagnose LLM backend issues.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import (
    LLM_BACKEND, GOOGLE_API_KEY, GROQ_API_KEY,
    GEMINI_MODEL, GROQ_MODEL,
    GEMINI_TIMEOUT, GROQ_TIMEOUT
)
from generator import check_package_installed, print_configuration_diagnostics

def test_configuration():
    """Test LLM configuration without making API calls."""
    print("=" * 60)
    print("LLM CONFIGURATION TEST")
    print("=" * 60)
    
    # Print diagnostics
    print_configuration_diagnostics()
    
    # Check package installation
    gemini_installed = check_package_installed("google.genai")
    groq_installed = check_package_installed("groq")
    
    print("\n" + "=" * 60)
    print("CONFIGURATION STATUS")
    print("=" * 60)
    
    # Check API keys
    gemini_key_configured = bool(GOOGLE_API_KEY and GOOGLE_API_KEY.strip())
    groq_key_configured = bool(GROQ_API_KEY and GROQ_API_KEY.strip())
    
    if not gemini_installed:
        print("❌ ERROR: google.genai package not installed")
        print("   Run: venv\\Scripts\\pip install google-genai")
    else:
        print("✅ google.genai package installed")
    
    if not groq_installed:
        print("❌ ERROR: groq package not installed")
        print("   Run: venv\\Scripts\\pip install groq")
    else:
        print("✅ groq package installed")
    
    if not gemini_key_configured:
        print("❌ ERROR: GOOGLE_API_KEY not configured in .env")
        print("   Add: GOOGLE_API_KEY=your_key_here to .env file")
    else:
        print("✅ GOOGLE_API_KEY configured")
    
    if not groq_key_configured:
        print("⚠️  WARNING: GROQ_API_KEY not configured in .env")
        print("   Add: GROQ_API_KEY=your_key_here to .env file (optional)")
    else:
        print("✅ GROQ_API_KEY configured")
    
    print(f"\nSelected backend: {LLM_BACKEND}")
    print(f"Gemini model: {GEMINI_MODEL}")
    print(f"Groq model: {GROQ_MODEL}")
    
    # Check if selected backend is available
    if LLM_BACKEND == "gemini":
        if not gemini_key_configured:
            print("\n❌ CRITICAL: Gemini selected but API key not configured")
            print("   Please configure GOOGLE_API_KEY in .env file")
        elif not gemini_installed:
            print("\n❌ CRITICAL: Gemini selected but package not installed")
            print("   Run: venv\\Scripts\\pip install google-genai")
        else:
            print("\n✅ Gemini backend should be available")
    elif LLM_BACKEND == "groq":
        if not groq_key_configured:
            print("\n❌ CRITICAL: Groq selected but API key not configured")
            print("   Please configure GROQ_API_KEY in .env file")
        elif not groq_installed:
            print("\n❌ CRITICAL: Groq selected but package not installed")
            print("   Run: venv\\Scripts\\pip install groq")
        else:
            print("\n✅ Groq backend should be available")
    
    print("=" * 60)

if __name__ == "__main__":
    test_configuration()
