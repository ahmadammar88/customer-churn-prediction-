#!/bin/bash

# Update system
sudo yum update -y

# Install Python and pip
sudo yum install python3 python3-pip -y

# Install required packages
pip3 install -r requirements.txt

# Install screen (for running the app in background)
sudo yum install screen -y

# Create a screen session for the app
screen -dmS streamlit

# Send commands to the screen session
screen -S streamlit -X stuff "streamlit run app.py --server.port 8504 --server.address 0.0.0.0\n"

# Start MLflow server in a new screen session
screen -dmS mlflow
screen -S mlflow -X stuff "python -m mlflow ui --host 0.0.0.0\n" 