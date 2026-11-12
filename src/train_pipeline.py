import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from imblearn.over_sampling import SMOTE

def run_ml_pipeline():
    print("[INFO] Loading cleaned processed dataset...")
    processed_path = "data/processed/borrowers_cleaned.csv"
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed data not found at {processed_path}. Run clean_validate.py first.")
        
    df = pd.read_csv(processed_path)
    
    # 1. Feature Engineering
    print("[INFO] Performing feature engineering...")
    # Add mock loan purpose notes for TF-IDF vectorization
    np.random.seed(42)
    purposes = [
        "Need funds for business expansion and inventory",
        "Personal medical emergency and hospital bills",
        "Home renovation and kitchen remodeling",
        "Consolidating high-interest credit card debt",
        "Higher education tuition fee payment",
        "Purchasing a new vehicle for commute"
    ]
    df['loan_purpose_notes'] = np.random.choice(purposes, size=len(df))
    
    # Datetime feature extraction (simulating application timestamp)
    base_date = pd.Timestamp("2026-01-01")
    df['application_date'] = [base_date + pd.Timedelta(days=int(i*15)) for i in range(len(df))]
    df['app_day_of_week'] = pd.to_datetime(df['application_date']).dt.day_of_week
    
    # 3-loan rolling average simulation (using late_count or income)
    df['income_rolling_avg'] = df['annual_income'].rolling(window=3, min_periods=1).mean()
    
    # Log transform skew-heavy income
    df['log_annual_income'] = np.log1p(df['annual_income'])
    
    # Define features (X) and target (y)
    X = df[['credit_score', 'late_count', 'total_payments', 'app_day_of_week', 'income_rolling_avg', 'log_annual_income', 'city', 'loan_purpose_notes']]
    y = df['is_default']
    
    # Train-test split (stratified due to ~18% minority class)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 2. Preprocessing ColumnTransformer
    numeric_features = ['credit_score', 'late_count', 'total_payments', 'app_day_of_week', 'income_rolling_avg', 'log_annual_income']
    categorical_features = ['city']
    text_features = 'loan_purpose_notes'
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features),
            ('text', TfidfVectorizer(max_features=50), text_features)
        ]
    )
    
    # Apply SMOTE on training data (preventing data leakage by transforming inside pipeline structure)
    print("[INFO] Applying SMOTE for class imbalance handling...")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Convert sparse matrix back or apply SMOTE directly
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)
    print(f"Original train shape: {X_train_processed.shape}, SMOTE train shape: {X_train_smote.shape}")
    
    # 3. Model Training & Experiment Tracking
    models = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    experiment_results = []
    best_score = 0.0
    best_model_name = ""
    best_pipeline_obj = None
    
    X_test_processed = preprocessor.transform(X_test)
    
    for name, model in models.items():
        print(f"[INFO] Training {name}...")
        model.fit(X_train_smote, y_train_smote)
        
        y_pred = model.predict(X_test_processed)
        y_proba = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = 0.5
            
        metrics = {
            "model_name": name,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4)
        }
        experiment_results.append(metrics)
        
        # Track best model based on F1-score
        if f1 >= best_score:
            best_score = f1
            best_model_name = name
            best_pipeline_obj = (preprocessor, model)
            
    # Save experiment tracking JSON
    tracking_path = "reports/experiment_metrics.json"
    with open(tracking_path, "w") as f:
        json.dump(experiment_results, f, indent=4)
    print(f"[SUCCESS] Experiment metrics logged to {tracking_path}")
    
    # Save best versioned model bundle
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"models/credit_risk_model_{timestamp}.pkl"
    joblib.dump(best_pipeline_obj, model_filename)
    joblib.dump(best_pipeline_obj, "models/credit_risk_model_latest.pkl")
    print(f"[SUCCESS] Best model ({best_model_name}) saved to {model_filename}")
    
    # 4. Sample Inference Test
    print("\n--- 4. SAMPLE INFERENCE TEST ---")
    sample_input = X_test.iloc[[0]].copy()
    print("Sample Borrower Features:")
    print(sample_input.to_dict(orient="records")[0])
    
    sample_processed = best_pipeline_obj[0].transform(sample_input)
    prediction = best_pipeline_obj[1].predict(sample_processed)[0]
    probability = best_pipeline_obj[1].predict_proba(sample_processed)[0][1]
    
    print(f"[INFERENCE RESULT] Predicted Default Class: {int(prediction)} (Probability: {round(float(probability), 4)})")

if __name__ == "__main__":
    run_ml_pipeline()
