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
            print(f"✓ {package} is installed")
        except ImportError:
            missing.append(package)
            print(f"✗ {package} is NOT installed")
    
    return missing

def install_dependencies():
    """Install missing dependencies."""
    print("\n📦 Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✓ Dependencies installed successfully!\n")

def run_app():
    """Run the Streamlit application."""
    print("\n" + "="*60)
    print("🚀 Starting LIS Solver - Professional Edition")
    print("="*60)
    print("\n📍 The app will open in your browser...")
    print("🌐 URL: http://localhost:8501")
    print("\n💡 Tips:")
    print("   • Type 'q' and press Enter to stop the server")
    print("   • The app auto-reloads when you modify the code")
    print("\n" + "="*60 + "\n")
    
    # Run Streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])

def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("🎯 LIS Solver - Professional Edition Setup")
    print("="*60 + "\n")
    
    # Check dependencies
    print("📋 Checking dependencies...\n")
    missing = check_dependencies()
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        response = input("\nWould you like to install missing dependencies? (y/n): ").strip().lower()
        if response == 'y':
            install_dependencies()
        else:
            print("❌ Cannot run without required dependencies. Exiting.")
            return
    else:
        print("\n✅ All dependencies are installed!")
    
    # Run app
    try:
        run_app()
    except KeyboardInterrupt:
        print("\n\n✋ Server stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
