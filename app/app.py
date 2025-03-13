from flask import Flask, request, render_template, jsonify
import mlflow
import pandas as pd
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import DataPreprocessor

app = Flask(__name__)

def get_feature_names():
    """Get the feature names in the correct order"""
    with open('models/feature_names.txt', 'r') as f:
        return [line.strip() for line in f.readlines()]

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html', prediction=None, error=None)
        
    try:
        # Get form data
        data = {
            'Age': float(request.form['Age']),
            'Occupation': request.form['Occupation'],
            'Marital_Status': request.form['Marital_Status'],
            'Policies_Purchased': float(request.form['Policies_Purchased']),
            'Policy_Lapses': float(request.form['Policy_Lapses']),
            'Claim_Frequency': float(request.form['Claim_Frequency']),
            'Claim_Amount': float(request.form['Claim_Amount']),
            'Premium_Paid': float(request.form['Premium_Paid']),
            'Renewal_History': request.form['Renewal_History'],
            'Payment_Mode': request.form['Payment_Mode'],
            'Payment_Timing': request.form['Payment_Timing'],
            'Customer_Service_Interactions': float(request.form['Customer_Service_Interactions']),
            'Response_to_Offers': request.form['Response_to_Offers'],
            'Cross_Selling_Products': request.form['Cross_Selling_Products']
        }
        
        print("Received data:", data)
            
        # Convert to DataFrame with correct feature order
        df = pd.DataFrame([data])
        feature_names = get_feature_names()
        df = df[feature_names]
        
        # Load and use preprocessor
        preprocessor = DataPreprocessor()
        preprocessor.load('models/preprocessor.pkl')
        df_processed = preprocessor.preprocess(df, is_training=False)
        
        # Load the local model
        model = mlflow.sklearn.load_model("models/model")
        
        # Make prediction
        prediction = float(model.predict(df_processed)[0])
        print("Prediction:", prediction)
        
        return render_template('index.html', prediction=prediction, error=None)
    
    except Exception as e:
        print("Error:", str(e))
        return render_template('index.html', prediction=None, error=f"Error making prediction: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)