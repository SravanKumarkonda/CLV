# Customer Lifetime Value Prediction Project

## Overview
This project predicts customer lifetime value (CLV) using machine learning. It uses customer data including demographics, policy information, and interaction history to predict the potential lifetime value of insurance customers.

## Data Description
The dataset includes the following features:
- Age: Customer's age
- Occupation: Customer's profession (Engineer, Doctor, Business, etc.)
- Marital_Status: Customer's marital status
- Policies_Purchased: Number of policies
- Policy_Lapses: Number of policy lapses
- Claim_Frequency: Frequency of claims
- Claim_Amount: Total claim amount
- Premium_Paid: Premium amount paid
- Renewal_History: History of policy renewals
- Payment_Mode: Mode of payment
- Payment_Timing: Timing of payments
- Customer_Service_Interactions: Number of customer service interactions
- Response_to_Offers: Customer's response to offers
- Cross_Selling_Products: Types of products purchased

## Technical Implementation

### Preprocessing
- Label encoding for categorical variables:
  - Occupation
  - Marital_Status
  - Renewal_History
  - Payment_Mode
  - Payment_Timing
  - Response_to_Offers
  - Cross_Selling_Products
- Standard scaling for numerical features

### Model
- Algorithm: XGBoost Regressor
- Hyperparameter Optimization using Optuna:
  - max_depth: 3-10
  - learning_rate: 0.01-0.3
  - n_estimators: 100-1000
  - min_child_weight: 1-7
  - subsample: 0.6-1.0
  - colsample_bytree: 0.6-1.0

### MLflow Integration
- Experiment tracking
- Model versioning
- Parameter logging
- Metric tracking

## Setup Instructions

### Prerequisites
- Docker Desktop
- Git

### Installation Steps

1. Clone the repository:
git clone https://github.com/SravanKumarkonda/CLV.git

                     or

unzip CLV.zip

2. Navigate to the project directory:
cd path/to/CLV folder   

3. Build and start the Docker containers:
docker-compose up --build

4. In a new terminal, train the model:
docker exec -it clv-webapp-1 python src/train.py


4. Access the web interface:
   - Open your browser
   - Go to http://localhost:8000
   - Enter customer details to get CLV prediction
   - Go to http://localhost:5000  # for mlflow


### Making Predictions
1. Fill in all required fields in the web form
2. Click "Predict CLV"
3. View the predicted Customer Lifetime Value

## Technologies Used
- Python 3.8+
- Flask for web interface
- XGBoost for machine learning
- MLflow for experiment tracking
- Docker for containerization
- Pandas for data manipulation
- Scikit-learn for preprocessing
- Optuna for hyperparameter optimization



