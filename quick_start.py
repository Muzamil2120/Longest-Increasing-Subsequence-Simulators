"""
Quick Start Script for LIS Solver - Professional Edition

This script sets up and runs the beautiful LIS Solver application.
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if all required packages are installed."""
    required = ['streamlit', 'plotly', 'numpy', 'matplotlib']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package} is installed")
        except ImportError:
            missing.append(package)
            print(f"  ✗ {package} is NOT installed")
    
    return missing

def install_dependencies():
    """Install missing dependencies."""
    print("\n" + "─" * 60)
    print("📦 Installing dependencies...")
    print("─" * 60 + "\n")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("\n  ✓ Dependencies installed successfully!")
    print("─" * 60 + "\n")

def run_app():
    """Run the Streamlit application."""
    print("\n" + "═" * 60)
    print("🚀 Starting LIS Solver - Professional Edition")
    print("═" * 60)
    print("""
  📍 App Information:
    • Local URL:    http://localhost:8501
    • Network URL:  http://192.168.1.14:8501
    
  💡 Quick Tips:
    • Type 'q' and press Enter to stop the server
    • The app auto-reloads when you modify files
    
  ⚙️  Features:
    • Interactive LIS visualization
    • Step-by-step algorithm trace
    • Performance benchmarking
    • Multiple test cases
""")
    print("═" * 60 + "\n")
    
    # Run Streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])

def main():
    """Main entry point."""
    # Header Card
    print("\n" + "┏" + "━" * 58 + "┓")
    print("┃" + " " * 10 + "🎯 LIS Solver - Professional Edition" + " " * 11 + "┃")
    print("┗" + "━" * 58 + "┛\n")
    
    # Dependencies Check Card
    print("┌" + "─" * 58 + "┐")
    print("│  📋 Checking Dependencies..." + " " * 28 + "│")
    print("├" + "─" * 58 + "┤")
    missing = check_dependencies()
    print("└" + "─" * 58 + "┘")
    
    # Installation Card (if needed)
    if missing:
        print(f"\n⚠️  Missing: {', '.join(missing)}")
        response = input("\n  Install missing packages? (y/n): ").strip().lower()
        if response == 'y':
            install_dependencies()
        else:
            print("\n  ❌ Cannot run without required dependencies.")
            return
    else:
        print("\n┌" + "─" * 58 + "┐")
        print("│  ✅ All dependencies installed successfully!       │")
        print("└" + "─" * 58 + "┘\n")
    
    # Run app
    try:
        run_app()
    except KeyboardInterrupt:
        print("\n\n" + "┌" + "─" * 58 + "┐")
        print("│  ✋ Server stopped. Goodbye!                          │")
        print("└" + "─" * 58 + "┘\n")
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
