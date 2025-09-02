#!/usr/bin/env python3
"""
Quick test script to verify the Agentic RAG setup
"""

import os
import sys

def test_setup():
    """Test if everything is set up correctly"""
    
    print("🧪 AGENTIC RAG SETUP TEST")
    print("="*50)
    
    # Test 1: Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        print("✅ GOOGLE_API_KEY is set")
        print(f"   Key preview: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else ''}")
    else:
        print("❌ GOOGLE_API_KEY is not set")
        print("💡 Set it with: $env:GOOGLE_API_KEY=\"your_api_key_here\"")
        return False
    
    # Test 2: Check main file
    try:
        print("🔍 Testing main system import...")
        import agentic_rag
        print("✅ Main system import successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("💡 Check that agentic_rag.py exists and is not corrupted")
        return False
    except Exception as e:
        print(f"⚠️ Import warning: {e}")
    
    print("\n🎉 BASIC TESTS PASSED!")
    print("💡 You can now run: python agentic_rag.py")
    return True

if __name__ == "__main__":
    if not test_setup():
        sys.exit(1)
