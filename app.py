import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import os

# Set environment variables for Azure
os.environ['STREAMLIT_SERVER_PORT'] = '8000'
os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Cache the model loading
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_model():
    try:
        model = joblib.load('churn_model.joblib')
        encoders = joblib.load('churn_encoders.joblib')
        return model, encoders
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model and encoders
model, encoders = load_model()

if model is None or encoders is None:
    st.error("Please run train_model.py first to create the model")
    st.stop()

# Streamlit UI
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict if they are likely to churn.")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1000.0)

with col2:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Cache the prediction function
@st.cache_data(ttl=300)  # Cache for 5 minutes
def make_prediction(model, encoders, input_data):
    """Make prediction using the trained model"""
    try:
        # Prepare input data
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical features
        input_df['Contract'] = encoders['contract'].transform(input_df['Contract'])
        input_df['PaymentMethod'] = encoders['payment'].transform(input_df['PaymentMethod'])
        input_df['InternetService'] = encoders['internet'].transform(input_df['InternetService'])
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # Convert prediction to Yes/No using the target encoder
        return encoders['target'].inverse_transform(prediction)[0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# Add a predict button
if st.button("Predict Churn"):
    # Prepare input data
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'PaymentMethod': payment_method,
        'InternetService': internet_service
    }
    
    # Make prediction using the loaded model
    prediction = make_prediction(model, encoders, input_data)
    
    # Display prediction
    st.subheader("Prediction Result")
    if prediction is not None:
        st.write(prediction)

# Add some information about the model
st.sidebar.header("About")
st.sidebar.info("""
This application predicts customer churn based on various factors including:
- Tenure
- Monthly Charges
- Total Charges
- Contract Type
- Payment Method
- Internet Service

The model is trained on historical customer data and uses Random Forest algorithm for predictions.
""")

# Expose the server for gunicorn
server = st.server 