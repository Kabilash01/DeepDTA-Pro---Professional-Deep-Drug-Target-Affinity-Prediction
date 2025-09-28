"""
Launch script for DeepDTA-Pro web application
Run this file to start the web interface
"""

import subprocess
import sys
import os
from pathlib import Path

def install_streamlit():
    """Install Streamlit if not available"""
    try:
        import streamlit
        print("✅ Streamlit already installed")
    except ImportError:
        print("📦 Installing Streamlit...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
        print("✅ Streamlit installed successfully")

def main():
    """Launch the web application"""
    # Install Streamlit if needed
    install_streamlit()
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print("🚀 Starting DeepDTA-Pro Web Interface...")
    print("📱 The web app will open in your browser automatically")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    
    # Launch Streamlit app
    cmd = [sys.executable, "-m", "streamlit", "run", "web_app.py", "--server.port=8501"]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()