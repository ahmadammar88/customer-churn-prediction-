import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import mlflow
import mlflow.sklearn

def prepare_data():
    """Prepare the data for training"""
    # Load data
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing values
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
    
    # Select features
    features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'PaymentMethod', 'InternetService']
    X = df[features]
    y = df['Churn']
    
    # Create encoders for each categorical feature
    encoders = {
        'contract': LabelEncoder(),
        'payment': LabelEncoder(),
        'internet': LabelEncoder(),
        'target': LabelEncoder()
    }
    
    # Encode categorical features
    X['Contract'] = encoders['contract'].fit_transform(X['Contract'])
    X['PaymentMethod'] = encoders['payment'].fit_transform(X['PaymentMethod'])
    X['InternetService'] = encoders['internet'].fit_transform(X['InternetService'])
    
    # Encode target variable (Yes=1, No=0)
    y = encoders['target'].fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, encoders

def train_and_save_model():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Customer_Churn_Prediction")
    
    with mlflow.start_run(run_name="RandomForest_Model"):
        # Prepare data
        X_train, X_test, y_train, y_test, encoders = prepare_data()
        
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
        
        # Log parameters
        mlflow.log_params({
            "n_estimators": 100,
            "random_state": 42,
            "test_size": 0.2
        })
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        })
        mlflow.log_text(feature_importance.to_string(), "feature_importance.txt")
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model and encoders locally
        joblib.dump(model, 'churn_model.joblib')
        joblib.dump(encoders, 'churn_encoders.joblib')
        
        print("Model and encoders saved successfully!")
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    train_and_save_model() 