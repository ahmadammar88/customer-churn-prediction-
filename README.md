# Customer Churn Prediction App

A Streamlit application that predicts customer churn using machine learning. The application uses a Random Forest model trained on the Telco Customer Churn dataset.

## Features

- Interactive input form for customer details
- Real-time churn prediction
- Model performance metrics
- MLflow integration for experiment tracking
- AWS Amplify deployment ready

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. (Optional) Start MLflow UI for experiment tracking:
```bash
python -m mlflow ui --port 5000
```

## Usage

1. Open the application in your browser (default: http://localhost:8501)
2. Enter customer details in the input fields
3. Click "Predict Churn" to get the prediction
4. View model performance metrics

## AWS Amplify Deployment

This project is configured for deployment on AWS Amplify. The main branch contains the production-ready code.

## Project Structure

- `app.py`: Main Streamlit application
- `requirements.txt`: Python dependencies
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: Dataset
- `.gitignore`: Git ignore rules
- `README.md`: Project documentation 