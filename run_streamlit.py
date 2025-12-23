#!/usr/bin/env python
"""
Streamlit startup script for LIS Project
Run this to launch the Streamlit web UI
"""

import os
import subprocess
import sys

if __name__ == "__main__":
    print("=" * 70)
    print("Starting Streamlit LIS Solver UI...")
    print("=" * 70)
    print("\nThe app will open in your browser at: http://localhost:8501")
    print("Press Ctrl+C to stop the server\n")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501"]

    try:
        completed = subprocess.run(cmd, cwd=project_root)
    except KeyboardInterrupt:
        print("\n\nStreamlit server stopped.")
        sys.exit(0)

    # Streamlit sometimes returns non-zero when stopped; keep the message friendly.
    if completed.returncode not in (0, None):
        print("\nStreamlit exited with a non-zero status.")
        print("If the browser didn't open, try visiting: http://localhost:8501")
        print("If the port is busy, close other Streamlit terminals or change the port.")

    sys.exit(completed.returncode or 0)
