#!/bin/bash
echo "🚀 Starting Enhanced NAVUS on Render..."

# Install Python dependencies
pip install -r ../requirements-production.txt

# Generate the massive dataset if it doesn't exist
if [ ! -f "../Training/massive_navus_dataset_latest.json" ]; then
    echo "📊 Generating training dataset..."
    cd ../Scripts
    python generate_massive_dataset.py
    cd ../WebApp
fi

# Start the Streamlit app
streamlit run enhanced_backend.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true