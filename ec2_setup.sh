#!/bin/bash

# Update system
sudo yum update -y

# Install Git
sudo yum install git -y

# Install Python and pip
sudo yum install python3 python3-pip -y

# Install screen (for running the app in background)
sudo yum install screen -y

# Clone your repository
git clone https://github.com/ahmadammar88/customer-churn-prediction-.git
cd customer-churn-prediction-

# Install required packages
pip3 install -r requirements.txt

# Train the model
python3 train_model.py

# Create a screen session for the app
screen -dmS streamlit

# Send commands to the screen session
screen -S streamlit -X stuff "streamlit run app.py --server.port 8504 --server.address 0.0.0.0\n"

# Start MLflow server in a new screen session
screen -dmS mlflow
screen -S mlflow -X stuff "python3 -m mlflow ui --host 0.0.0.0\n" 