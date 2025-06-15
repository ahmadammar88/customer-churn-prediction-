# Customer Churn Prediction App

A Streamlit application for predicting customer churn using machine learning. The app uses a Random Forest model trained on the Telco Customer Churn dataset.

## Features

- Interactive input form for customer details
- Real-time churn prediction (Yes/No)
- MLflow integration for model tracking
- Simple and clean user interface

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/ahmadammar88/customer-churn-prediction-.git
cd customer-churn-prediction-
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python train_model.py
```

4. Run the Streamlit app:
```bash
streamlit run app.py --server.port 8502
```

## Optional: MLflow Tracking

To view model performance metrics:

1. Start MLflow server:
```bash
python -m mlflow ui
```

2. Open http://127.0.0.1:5000 in your browser

## Project Structure

- `app.py`: Streamlit application
- `train_model.py`: Model training script
- `requirements.txt`: Project dependencies
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: Dataset

## Usage

1. Enter customer details in the input form:
   - Tenure (months)
   - Monthly Charges
   - Total Charges
   - Contract Type
   - Payment Method
   - Internet Service

2. Click "Predict Churn" to get the prediction

3. The result will show either "Yes" or "No" for churn prediction 