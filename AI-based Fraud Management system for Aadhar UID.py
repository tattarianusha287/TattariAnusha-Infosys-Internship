# AI-based Fraud Management System for Aadhaar UID
# =================================================

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime, timedelta
import json
import hashlib
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fraud_detection.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("AadhaarFraudDetection")


# =============================
# Data Anonymization Functions
# =============================

def hash_aadhaar(aadhaar_number):
    """
    Hash Aadhaar number for privacy preservation
    Only for storing in data - authentication will use proper encryption methods
    """
    # In production, use more secure methods with proper salt and cryptographic algorithms
    return hashlib.sha256(str(aadhaar_number).encode()).hexdigest()


def mask_aadhaar(aadhaar_number):
    """Mask Aadhaar number for display purposes"""
    if not aadhaar_number or len(str(aadhaar_number)) != 12:
        return "Invalid-Aadhaar"
    return "XXXX-XXXX-" + str(aadhaar_number)[-4:]


def anonymize_data(df, sensitive_columns):
    """Anonymize sensitive data before processing"""
    df_anonymized = df.copy()
    
    for column in sensitive_columns:
        if column == 'aadhaar_number':
            df_anonymized[column] = df_anonymized[column].apply(hash_aadhaar)
        elif 'name' in column:
            df_anonymized[column] = 'ANONYMIZED'
        elif 'address' in column:
            df_anonymized[column] = 'ANONYMIZED'
        elif 'phone' in column:
            df_anonymized[column] = df_anonymized[column].apply(lambda x: 'XXXXXX' + str(x)[-4:] if pd.notnull(x) else x)
            
    return df_anonymized


# ===========================
# Feature Engineering Class
# ===========================

class AadhaarFeatureEngineering:
    def _init_(self):
        self.scaler = StandardScaler()
        
    def extract_features(self, df):
        """
        Extract features from raw authentication data
        """
        # Create a copy to avoid modifying the original
        data = df.copy()
        
        # Log the process
        logger.info(f"Starting feature extraction on dataset with {len(data)} rows")
        
        # Time-based features
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
            data['month'] = data['timestamp'].dt.month
        
        # Location-based features
        if all(col in data.columns for col in ['auth_latitude', 'auth_longitude']):
            # Calculate distance from usual location if available
            if all(col in data.columns for col in ['usual_latitude', 'usual_longitude']):
                data['location_distance'] = np.sqrt(
                    (data['auth_latitude'] - data['usual_latitude'])**2 + 
                    (data['auth_longitude'] - data['usual_longitude'])**2
                )
            
            # Flag authentication attempts from unusual locations
            if 'location_distance' in data.columns:
                data['unusual_location'] = data['location_distance'].apply(
                    lambda x: 1 if x > 0.5 else 0  # Simple threshold, adjust as needed
                )
        
        # Device and network features
        if 'device_id' in data.columns:
            data['new_device'] = data.groupby('aadhaar_number')['device_id'].transform(
                lambda x: x != x.mode()[0] if not x.mode().empty else 1
            ).astype(int)
        
        if 'ip_address' in data.columns:
            data['ip_change'] = data.groupby('aadhaar_number')['ip_address'].transform(
                lambda x: x != x.shift(1)
            ).fillna(0).astype(int)
        
        # Authentication pattern features
        if 'aadhaar_number' in data.columns and 'timestamp' in data.columns:
            # Number of authentication attempts in last day/week
            data['auth_attempts_day'] = data.groupby('aadhaar_number')['timestamp'].transform(
                lambda x: x.between(x.max() - pd.Timedelta(days=1), x.max()).sum()
            )
            
            # Time since last authentication
            data['time_since_last_auth'] = data.sort_values('timestamp').groupby('aadhaar_number')['timestamp'].diff()
            data['time_since_last_auth'] = data['time_since_last_auth'].dt.total_seconds() / 3600  # Convert to hours
            
            # Authentication method changes
            if 'auth_method' in data.columns:
                data['auth_method_change'] = data.sort_values('timestamp').groupby('aadhaar_number')['auth_method'].transform(
                    lambda x: (x != x.shift(1)).astype(int)
                ).fillna(0).astype(int)
        
        # Authentication failure features
        if 'auth_success' in data.columns:
            data['auth_success'] = data['auth_success'].astype(int)
            data['recent_failures'] = data.sort_values('timestamp').groupby('aadhaar_number')['auth_success'].transform(
                lambda x: (1 - x).rolling(window=3, min_periods=1).sum()
            )
        
        # Drop columns that are no longer needed or shouldn't be used for modeling
        columns_to_drop = ['timestamp', 'device_id', 'ip_address', 'auth_latitude', 
                          'auth_longitude', 'usual_latitude', 'usual_longitude']
        
        for col in columns_to_drop:
            if col in data.columns:
                data = data.drop(col, axis=1)
        
        # Fill missing values with appropriate methods
        numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = data.select_dtypes(include=['object', 'category']).columns
        
        # For this example, we'll simply drop rows with missing values in the target
        if 'is_fraud' in data.columns:
            data = data.dropna(subset=['is_fraud'])
            
        # Return the processed data
        logger.info(f"Feature extraction completed. Final dataset shape: {data.shape}")
        return data
        
    def preprocess_data(self, data, target_column='is_fraud', train=True):
        """
        Preprocess data for ML model training or prediction
        """
        logger.info(f"Preprocessing data with shape {data.shape}")
        
        # Separate features and target
        if target_column in data.columns:
            X = data.drop(target_column, axis=1)
            y = data[target_column]
        else:
            X = data.copy()
            y = None
            
        # Identify numeric and categorical features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Fit or transform data
        if train:
            X_processed = preprocessor.fit_transform(X)
            # Save the preprocessor
            joblib.dump(preprocessor, 'aadhaar_preprocessor.pkl')
        else:
            # Load saved preprocessor
            if os.path.exists('aadhaar_preprocessor.pkl'):
                preprocessor = joblib.load('aadhaar_preprocessor.pkl')
                X_processed = preprocessor.transform(X)
            else:
                logger.error("No preprocessor found. Please train the model first.")
                return None, None
                
        logger.info(f"Preprocessing completed. Processed features shape: {X_processed.shape}")
        return X_processed, y


# ===========================
# Fraud Detection Model
# ===========================

class AadhaarFraudDetectionModel:
    def _init_(self, model_type='random_forest'):
        """
        Initialize the fraud detection model
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('random_forest' or 'gradient_boosting')
        """
        self.model_type = model_type
        self.feature_engineering = AadhaarFeatureEngineering()
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.model_path = f"aadhaar_fraud_{model_type}.pkl"
        logger.info(f"Initialized {model_type} model for fraud detection")
            
    def train(self, data, target_column='is_fraud', perform_grid_search=False):
        """
        Train the fraud detection model
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input dataset with features and target
        target_column : str
            Name of the target column
        perform_grid_search : bool
            Whether to perform grid search for hyperparameter tuning
        """
        logger.info("Starting model training process")
        
        # Extract features
        data_processed = self.feature_engineering.extract_features(data)
        
        # Preprocess data
        X, y = self.feature_engineering.preprocess_data(data_processed, target_column, train=True)
        
        if X is None or y is None:
            logger.error("Error in preprocessing data")
            return False
            
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Perform grid search if specified
        if perform_grid_search:
            logger.info("Performing grid search for hyperparameter tuning")
            
            if self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            else:  # gradient_boosting
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
                
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5, scoring='f1', n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
        else:
            # Train the model
            self.model.fit(X_train, y_train)
            
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        
        # Print classification report
        report = classification_report(y_test, y_pred)
        logger.info(f"Model evaluation:\n{report}")
        
        # Print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion matrix:\n{cm}")
        
        # Save the model
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        
        return True
        
    def predict(self, auth_data):
        """
        Predict fraud probability for new authentication data
        
        Parameters:
        -----------
        auth_data : pandas.DataFrame
            Authentication data
            
        Returns:
        --------
        pandas.DataFrame
            Original data with fraud probability and prediction
        """
        logger.info(f"Predicting fraud for {len(auth_data)} authentication records")
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            return None
            
        # Load the model
        self.model = joblib.load(self.model_path)
        
        # Extract features
        data_processed = self.feature_engineering.extract_features(auth_data)
        
        # Preprocess data
        X, _ = self.feature_engineering.preprocess_data(data_processed, train=False)
        
        if X is None:
            logger.error("Error in preprocessing data")
            return None
            
        # Make predictions
        fraud_proba = self.model.predict_proba(X)[:, 1]
        fraud_pred = self.model.predict(X)
        
        # Add predictions to original data
        result = auth_data.copy()
        result['fraud_probability'] = fraud_proba
        result['fraud_prediction'] = fraud_pred
        
        logger.info(f"Prediction completed. Found {fraud_pred.sum()} potential fraud cases")
        
        return result


# ===========================
# Real-time Alert System
# ===========================

class AadhaarAlertSystem:
    def _init_(self, threshold=0.7):
        """
        Initialize the alert system
        
        Parameters:
        -----------
        threshold : float
            Probability threshold for generating alerts
        """
        self.threshold = threshold
        self.alerts = []
        
    def generate_alerts(self, predictions):
        """
        Generate alerts for high-risk authentication attempts
        
        Parameters:
        -----------
        predictions : pandas.DataFrame
            DataFrame with fraud predictions and probabilities
            
        Returns:
        --------
        list
            List of alert dictionaries
        """
        alerts = []
        
        # Filter high-risk authentications
        high_risk = predictions[predictions['fraud_probability'] >= self.threshold]
        
        for _, row in high_risk.iterrows():
            alert = {
                'aadhaar': mask_aadhaar(row.get('aadhaar_number', 'Unknown')),
                'timestamp': str(row.get('timestamp', datetime.now())),
                'risk_score': float(row['fraud_probability']),
                'auth_method': row.get('auth_method', 'Unknown'),
                'location': f"{row.get('auth_latitude', 'Unknown')}, {row.get('auth_longitude', 'Unknown')}",
                'alert_id': hashlib.md5(f"{row.get('aadhaar_number', '')}{datetime.now()}".encode()).hexdigest()
            }
            
            alerts.append(alert)
            
        self.alerts.extend(alerts)
        logger.info(f"Generated {len(alerts)} alerts")
        
        return alerts
        
    def save_alerts(self, filename='aadhaar_alerts.json'):
        """Save alerts to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.alerts, f, indent=4)
        logger.info(f"Saved {len(self.alerts)} alerts to {filename}")
        
    def notify_security_team(self, alerts):
        """
        Send notifications to security team
        
        Parameters:
        -----------
        alerts : list
            List of alert dictionaries
        """
        # In a real implementation, this would send emails, SMS, or use a messaging system
        logger.info(f"Would notify security team about {len(alerts)} high-risk authentications")
        
        # Just log the alerts for this example
        for alert in alerts:
            logger.info(f"SECURITY ALERT: High risk authentication detected for {alert['aadhaar']} "
                       f"with risk score {alert['risk_score']:.2f}")


# ===========================
# API Interface
# ===========================

class AadhaarFraudAPI:
    def _init_(self):
        """Initialize the API interface"""
        self.model = AadhaarFraudDetectionModel(model_type='gradient_boosting')
        self.alert_system = AadhaarAlertSystem(threshold=0.7)
        
    def process_auth_request(self, auth_data):
        """
        Process authentication request
        
        Parameters:
        -----------
        auth_data : dict
            Authentication request data
            
        Returns:
        --------
        dict
            Processing result with risk assessment
        """
        # Convert to DataFrame
        df = pd.DataFrame([auth_data])
        
        # Anonymize sensitive data
        sensitive_columns = ['aadhaar_number', 'name', 'address', 'phone_number']
        df_anonymous = anonymize_data(df, sensitive_columns)
        
        # Make prediction
        prediction = self.model.predict(df_anonymous)
        
        if prediction is None:
            return {
                'status': 'error',
                'message': 'Failed to process authentication request'
            }
            
        # Generate alerts if needed
        if prediction['fraud_probability'].iloc[0] >= self.alert_system.threshold:
            alerts = self.alert_system.generate_alerts(prediction)
            self.alert_system.notify_security_team(alerts)
            
        # Return result
        return {
            'request_id': hashlib.md5(f"{auth_data.get('aadhaar_number', '')}{datetime.now()}".encode()).hexdigest(),
            'timestamp': datetime.now().isoformat(),
            'risk_score': float(prediction['fraud_probability'].iloc[0]),
            'recommendation': 'block' if prediction['fraud_prediction'].iloc[0] == 1 else 'allow',
            'additional_verification': prediction['fraud_probability'].iloc[0] >= 0.3
        }
        
    def batch_process(self, auth_data_list):
        """
        Process a batch of authentication requests
        
        Parameters:
        -----------
        auth_data_list : list
            List of authentication request dictionaries
            
        Returns:
        --------
        list
            List of processing results
        """
        results = []
        
        for auth_data in auth_data_list:
            result = self.process_auth_request(auth_data)
            results.append(result)
            
        return results


# ===========================
# Simulation and Testing
# ===========================

def generate_sample_data(n_samples=1000, fraud_ratio=0.1):
    """
    Generate sample data for testing
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    fraud_ratio : float
        Ratio of fraudulent transactions
    
    Returns:
    --------
    pandas.DataFrame
        Synthetic dataset
    """
    np.random.seed(42)
    
    # Generate Aadhaar numbers (12 digits)
    aadhaar_numbers = np.random.randint(100000000000, 999999999999, size=n_samples)
    
    # Generate timestamps over the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    timestamps = [start_date + timedelta(
        seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
    ) for _ in range(n_samples)]
    
    # Generate locations (latitude and longitude for India)
    latitudes = np.random.uniform(8.0, 37.0, n_samples)  # India's latitude range
    longitudes = np.random.uniform(68.0, 97.0, n_samples)  # India's longitude range
    
    # Generate usual locations (close to auth locations for non-fraud)
    usual_latitudes = np.zeros(n_samples)
    usual_longitudes = np.zeros(n_samples)
    
    # Authentication methods
    auth_methods = np.random.choice(
        ['fingerprint', 'iris', 'otp', 'face'], 
        size=n_samples, 
        p=[0.6, 0.2, 0.15, 0.05]
    )
    
    # Device IDs
    device_ids = [f"DEV-{np.random.randint(1000, 9999)}" for _ in range(n_samples)]
    
    # IP addresses
    ip_addresses = [f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}."
                   f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" 
                   for _ in range(n_samples)]
    
    # Generate fraud labels
    is_fraud = np.zeros(n_samples)
    fraud_indices = np.random.choice(
        range(n_samples), 
        size=int(n_samples * fraud_ratio), 
        replace=False
    )
    is_fraud[fraud_indices] = 1
    
    # For non-fraud cases, usual location is close to auth location
    for i in range(n_samples):
        if is_fraud[i] == 0:
            # Small random deviation for non-fraud
            usual_latitudes[i] = latitudes[i] + np.random.uniform(-0.01, 0.01)
            usual_longitudes[i] = longitudes[i] + np.random.uniform(-0.01, 0.01)
        else:
            # Larger deviation for fraud cases
            usual_latitudes[i] = latitudes[i] + np.random.uniform(-5, 5)
            usual_longitudes[i] = longitudes[i] + np.random.uniform(-5, 5)
            
            # Modify some patterns for fraud cases
            if np.random.random() < 0.7:
                # Unusual auth method for this user
                auth_methods[i] = np.random.choice(['fingerprint', 'iris', 'otp', 'face'])
            
            if np.random.random() < 0.8:
                # New device
                device_ids[i] = f"DEV-{np.random.randint(1000, 9999)}"
                
            if np.random.random() < 0.9:
                # Different IP
                ip_addresses[i] = f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}."
                f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
    
    # Create DataFrame
    df = pd.DataFrame({
        'aadhaar_number': aadhaar_numbers,
        'timestamp': timestamps,
        'auth_latitude': latitudes,
        'auth_longitude': longitudes,
        'usual_latitude': usual_latitudes,
        'usual_longitude': usual_longitudes,
        'auth_method': auth_methods,
        'device_id': device_ids,
        'ip_address': ip_addresses,
        'auth_success': np.random.choice([0, 1], size=n_samples, p=[0.05, 0.95]),
        'is_fraud': is_fraud
    })
    
    return df


def run_simulation():
    """Run a simulation to test the fraud detection system"""
    print("Starting AI-based Aadhaar Fraud Detection System simulation")
    
    # Generate sample data
    print("Generating synthetic data...")
    data = generate_sample_data(n_samples=10000, fraud_ratio=0.1)
    print(f"Generated {len(data)} records with {data['is_fraud'].sum()} fraud cases")
    
    # Initialize and train the model
    print("Training fraud detection model...")
    model = AadhaarFraudDetectionModel(model_type='gradient_boosting')
    model.train(data, perform_grid_search=True)
    
    # Initialize the API
    print("Initializing API and alert system...")
    api = AadhaarFraudAPI()
    
    # Test real-time processing with new data
    print("Testing real-time processing...")
    new_data = generate_sample_data(n_samples=100, fraud_ratio=0.2)
    
    # Process each record
    results = []
    for i, row in new_data.iterrows():
        auth_data = row.to_dict()
        result = api.process_auth_request(auth_data)
        results.append(result)
        
        if i < 5:  # Show first few results
            print(f"Request {i+1}:")
            print(f"  Risk score: {result['risk_score']:.4f}")
            print(f"  Recommendation: {result['recommendation']}")
            print(f"  Additional verification: {result['additional_verification']}")
            print()
    
    # Analyze results
    results_df = pd.DataFrame(results)
    print("\nSimulation results summary:")
    print(f"Total requests processed: {len(results_df)}")
    print(f"High-risk authentications: {(results_df['risk_score'] >= 0.7).sum()}")
    print(f"Medium-risk authentications: {((results_df['risk_score'] >= 0.3) & (results_df['risk_score'] < 0.7)).sum()}")
    print(f"Low-risk authentications: {(results_df['risk_score'] < 0.3).sum()}")
    
    print("\nFraud Detection System simulation completed")


# =============================
# Main Application Entry Point
# =============================

if _name_ == "_main_":
    run_simulation()