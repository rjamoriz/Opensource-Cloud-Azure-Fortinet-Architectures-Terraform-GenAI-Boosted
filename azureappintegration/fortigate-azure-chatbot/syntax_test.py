#!/usr/bin/env python3
"""
Quick syntax check for the problematic areas
"""

# Test the problematic st.info() section
import streamlit as st

def test_syntax():
    st.info("""
📋 **Microsoft Phi-1.5B**
- Parameters: 1.3 billion  
- Size: ~2.6GB
- Architecture: Transformer-based
- Optimized for code and reasoning tasks
""")
    
    print("Syntax check passed!")

if __name__ == "__main__":
    test_syntax()
