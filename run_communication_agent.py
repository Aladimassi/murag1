#!/usr/bin/env python3
"""
Simple test runner for the Communication Agent
This bypasses the MCP server and runs the agent directly
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Run the Communication Agent in interactive mode"""
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY environment variable not set.")
        print("💡 Please set your Gemini API key:")
        print("   For PowerShell: $env:GOOGLE_API_KEY=\"your_api_key_here\"")
        print("   For Command Prompt: set GOOGLE_API_KEY=your_api_key_here")
        print("   Or create a .env file with: GOOGLE_API_KEY=your_api_key_here")
        sys.exit(1)
    
    print("🤖 Communication Agent not yet restored in this version.")
    print("💡 Use the main agentic RAG system instead:")
    print("   python agentic_rag.py")

if __name__ == "__main__":
    main()
