#!/usr/bin/env python3
"""
Test script to demonstrate the dark mode toggle functionality
"""

import streamlit as st

def apply_theme_css():
    """Apply dark or light theme CSS based on session state"""
    if st.session_state.get('dark_mode', True):
        # Dark theme CSS
        theme_css = """
        <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        
        .stSidebar {
            background-color: #262730;
        }
        
        .stButton > button {
            background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(255, 107, 107, 0.4);
        }
        
        .stSelectbox > div > div {
            background-color: #262730;
            color: #fafafa;
        }
        
        .stTextInput > div > div > input {
            background-color: #262730;
            color: #fafafa;
            border: 1px solid #464852;
        }
        
        .main-header {
            background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1rem;
        }
        </style>
        """
    else:
        # Light theme CSS
        theme_css = """
        <style>
        .stApp {
            background-color: #ffffff;
            color: #262730;
        }
        
        .stSidebar {
            background-color: #f0f2f6;
        }
        
        .stButton > button {
            background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(255, 107, 107, 0.4);
        }
        
        .stSelectbox > div > div {
            background-color: #ffffff;
            color: #262730;
            border: 1px solid #e0e0e0;
        }
        
        .stTextInput > div > div > input {
            background-color: #ffffff;
            color: #262730;
            border: 1px solid #e0e0e0;
        }
        
        .main-header {
            background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1rem;
        }
        </style>
        """
    
    st.markdown(theme_css, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Dark Mode Toggle Demo",
        page_icon="🌙",
        layout="wide"
    )
    
    # Initialize dark mode session state
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = True  # Default to dark mode
    
    # Apply theme
    apply_theme_css()
    
    # Header with theme toggle
    col1, col2, col3 = st.columns([8, 1, 1])
    
    with col1:
        st.markdown('<h1 class="main-header">🌙 Dark Mode Toggle Demo</h1>', unsafe_allow_html=True)
    
    with col3:
        theme_icon = "🌙" if st.session_state.dark_mode else "☀️"
        theme_text = "Light Mode" if st.session_state.dark_mode else "Dark Mode"
        
        if st.button(f"{theme_icon}", 
                    help=f"Switch to {theme_text}",
                    key="theme_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    # Demo content
    st.write("This demonstrates the dark mode / light mode toggle functionality.")
    
    # Sidebar demo
    with st.sidebar:
        st.title("🔧 Settings")
        
        current_theme = "Dark Mode" if st.session_state.dark_mode else "Light Mode"
        st.info(f"Current Theme: **{current_theme}**")
        
        # Demo form elements
        st.selectbox("Demo Selectbox", ["Option 1", "Option 2", "Option 3"])
        st.text_input("Demo Text Input", "Type something here...")
        
        if st.button("Demo Button"):
            st.success("Button clicked!")
    
    # Main content
    st.subheader("Features Demonstrated:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Dark Mode Features:**
        - 🌙 Dark background (#0e1117)
        - ⚪ Light text (#fafafa)
        - 🎨 Gradient buttons
        - 📱 Responsive design
        """)
    
    with col2:
        st.markdown("""
        **Light Mode Features:**
        - ☀️ Light background (#ffffff)
        - ⚫ Dark text (#262730)
        - 🎨 Same gradient buttons
        - 📱 Responsive design
        """)
    
    st.info("Click the moon/sun icon in the top right corner to toggle between themes!")
    
    # Demo chat messages
    st.subheader("Demo Chat Interface:")
    
    with st.chat_message("user"):
        st.write("How does the dark mode look?")
    
    with st.chat_message("assistant"):
        st.write("The dark mode provides a sleek, modern interface that's easier on the eyes in low-light environments!")

if __name__ == "__main__":
    main()
