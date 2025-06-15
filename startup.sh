#!/bin/bash

# Create and activate virtual environment if it doesn't exist
if [ ! -d "env" ]; then
    python -m venv env
fi
source env/bin/activate

# Install required packages
pip install -r requirements.txt

# Train the model if not already trained
if [ ! -f "churn_model.pkl" ]; then
    python train_model.py
fi

# Set environment variables
export PYTHONUNBUFFERED=1
export STREAMLIT_SERVER_PORT=8000
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Start the Streamlit app
streamlit run app.py --server.port 8000 --server.address 0.0.0.0 