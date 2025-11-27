#!/bin/bash

# Script to run the 3D Fat Suppression Visualization App v2

echo "🧠 Starting 3D Fat Suppression Visualization App v2..."
echo "📍 App will be available at: http://localhost:8501"
echo "🔄 Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    python3 -m pip install streamlit pydicom
fi

# Run the Streamlit app
streamlit run visualize_brainFS_v2.py --server.port 8501 --server.address 0.0.0.0
