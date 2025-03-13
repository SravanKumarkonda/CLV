import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Updated with exact categories from your data
        self.known_categories = {
            'Occupation': ['Business', 'Doctor', 'Engineer', 'Retired', 'Self-Employed', 'Student', 'Teacher'],
            'Marital_Status': ['Divorced', 'Married', 'Single', 'Widowed'],
            'Renewal_History': ['Late', 'Missed', 'On-time'],
            'Payment_Mode': ['Bank Transfer', 'Cash', 'Credit Card', 'UPI'],
            'Payment_Timing': ['Late', 'On-time'],  # Fixed capitalization
            'Response_to_Offers': ['Negative', 'Neutral', 'Positive'],
            'Cross_Selling_Products': ['Auto + Home', 'Health + Life', 'Only Auto', 'Only Health', 'Only Home']  # Fixed categories
        }
        
    def preprocess(self, df, is_training=True):
        df_processed = df.copy()
        
        categorical_cols = list(self.known_categories.keys())
        numerical_cols = [
            'Age',
            'Policies_Purchased',
            'Policy_Lapses',
            'Claim_Frequency',
            'Claim_Amount',
            'Premium_Paid',
            'Customer_Service_Interactions'
        ]
        
        # Handle categorical columns
        for col in categorical_cols:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                # Fit on all known categories
                self.label_encoders[col].fit(self.known_categories[col])
            df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        # Handle numerical columns
        if is_training:
            df_processed[numerical_cols] = self.scaler.fit_transform(df_processed[numerical_cols])
        else:
            df_processed[numerical_cols] = self.scaler.transform(df_processed[numerical_cols])
            
        return df_processed
    
    def save(self, path):
        """Save preprocessor to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'known_categories': self.known_categories
            }, f)
    
    def load(self, path):
        """Load preprocessor from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.label_encoders = data['label_encoders']
            self.scaler = data['scaler']
            self.known_categories = data.get('known_categories', self.known_categories)