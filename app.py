import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# Set page config
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")

# Title and description
st.title("Customer Churn Predictor")
st.write("Enter customer details to predict if they are likely to churn.")

# Initialize MLflow with error handling
try:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Customer_Churn_Prediction")
    mlflow_available = True
except Exception as e:
    st.warning("MLflow server is not available. Running without experiment tracking.")
    mlflow_available = False

# Load and prepare the data
@st.cache_data
def load_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    return df

# Load the data
df = load_data()

# Create input fields
st.subheader("Customer Information")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=24)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1000.0)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Prepare the data for prediction
def prepare_data(df):
    # Select features
    features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'PaymentMethod', 'InternetService']
    X = df[features].copy()
    y = df['Churn'].copy()
    
    # Create separate label encoders for each categorical feature
    contract_encoder = LabelEncoder()
    payment_encoder = LabelEncoder()
    internet_encoder = LabelEncoder()
    target_encoder = LabelEncoder()
    
    # Handle categorical variables
    X['Contract'] = contract_encoder.fit_transform(X['Contract'])
    X['PaymentMethod'] = payment_encoder.fit_transform(X['PaymentMethod'])
    X['InternetService'] = internet_encoder.fit_transform(X['InternetService'])
    
    # Encode target variable
    y = target_encoder.fit_transform(y)
    
    # Handle missing values
    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
    X['TotalCharges'].fillna(X['TotalCharges'].mean(), inplace=True)
    
    encoders = {
        'contract': contract_encoder,
        'payment': payment_encoder,
        'internet': internet_encoder,
        'target': target_encoder
    }
    
    return X, y, encoders

# Train the model with MLflow tracking
@st.cache_resource
def train_model():
    X, y, encoders = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if mlflow_available:
        with mlflow.start_run(run_name="RandomForest_Model"):
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, pos_label=1),
                "recall": recall_score(y_test, y_pred, pos_label=1),
                "f1": f1_score(y_test, y_pred, pos_label=1),
                "roc_auc": roc_auc_score(y_test, y_pred_proba)
            }
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model parameters
            mlflow.log_params({
                "n_estimators": 100,
                "random_state": 42,
                "test_size": 0.2
            })
            
            # Log the model
            mlflow.sklearn.log_model(model, "model")
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            })
            mlflow.log_table(feature_importance, "feature_importance.json")
            
            return model, encoders, metrics
    else:
        # Train model without MLflow tracking
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, pos_label=1),
            "recall": recall_score(y_test, y_pred, pos_label=1),
            "f1": f1_score(y_test, y_pred, pos_label=1),
            "roc_auc": roc_auc_score(y_test, y_pred_proba)
        }
        
        return model, encoders, metrics

# Make prediction
def make_prediction(model, encoders, input_data):
    # Convert input data to match training data format
    input_df = pd.DataFrame([input_data])
    input_df['Contract'] = encoders['contract'].transform(input_df['Contract'])
    input_df['PaymentMethod'] = encoders['payment'].transform(input_df['PaymentMethod'])
    input_df['InternetService'] = encoders['internet'].transform(input_df['InternetService'])
    return model.predict_proba(input_df)[0]

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
    
    # Train model and make prediction
    model, encoders, metrics = train_model()
    prediction = make_prediction(model, encoders, input_data)
    
    # Display prediction
    st.subheader("Prediction Result")
    churn_probability = prediction[1] * 100
    st.write(f"Probability of customer churning: {churn_probability:.2f}%")
    
    if churn_probability > 50:
        st.error("High risk of churn! Consider taking action to retain this customer.")
    else:
        st.success("Low risk of churn. Customer is likely to stay.")
    
    # Display model performance metrics
    st.subheader("Model Performance Metrics")
    metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    st.table(metrics_df)

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

# Add MLflow UI link if available
if mlflow_available:
    st.sidebar.header("MLflow Tracking")
    st.sidebar.info("""
    To view detailed model metrics and performance:
    1. MLflow UI is running at http://localhost:5000
    2. Open the link in your browser to view experiment tracking
    """) 