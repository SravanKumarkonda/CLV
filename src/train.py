import os
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from preprocess import DataPreprocessor

def load_data():
    try:
        print("Starting to load data...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(os.path.dirname(current_dir), 'data', 'customer_data.csv')
        print(f"Looking for data file at: {data_path}")
        
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def train_model():
    print("Starting training process...")
    mlflow.set_tracking_uri("http://mlflow:5000")
    print("MLflow tracking URI set to: http://mlflow:5000")
    
    try:
        mlflow.create_experiment("CLV_Prediction")
    except:
        pass
    mlflow.set_experiment("CLV_Prediction")
    print("MLflow experiment set successfully")

    # Load and preprocess data
    df = load_data()
    if df is None:
        print("Failed to load data. Exiting...")
        return
    
    print("\nStarting preprocessing...")
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess(df, is_training=True)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save preprocessor
    preprocessor.save('models/preprocessor.pkl')
    print("Preprocessor saved successfully")
    
    # Split the data
    target_column = 'Customer_Lifetime_Value'
    X = df_processed.drop(target_column, axis=1)
    y = df_processed[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save feature names
    feature_names = list(X.columns)
    with open('models/feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    print("Feature names saved successfully")
    
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        return model.score(X_test, y_test)
    
    # Optimize hyperparameters
    print("Starting hyperparameter optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    
    # Train final model with best parameters
    print("Training final model with best parameters...")
    best_params = study.best_params
    final_model = XGBRegressor(**best_params)
    final_model.fit(X_train, y_train)
    
    # Save the model locally
    print("Saving the model...")
    model_path = os.path.join('models', 'model')
    mlflow.sklearn.save_model(final_model, model_path)
    print(f"Model saved successfully to {model_path}")
    
    # Log metrics
    final_score = final_model.score(X_test, y_test)
    print(f"Final R² score: {final_score}")

if __name__ == "__main__":
    print("Starting the training script...")
    train_model()