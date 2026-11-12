
import os
import pandas as pd
import numpy as np
import sweetviz as sv

def run_cleaning_pipeline():
    print("[INFO] Loading latest raw data snapshot...")
    raw_path = "data/raw/borrowers_raw_latest.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Run ingest.py first.")
        
    df = pd.read_csv(raw_path)
    
    # 1. Inspection
    print("--- 1. INITIAL INSPECTION ---")
    print(f"Shape: {df.shape}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    print(f"Duplicate Rows: {df.duplicated().sum()}")
    
    # 2. Deduplication & Missing Value Imputation
    df = df.drop_duplicates()
    
    # Impute numerical columns with median
    num_cols = ['annual_income', 'credit_score', 'late_count', 'total_payments']
    for col in num_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
            
    # Impute categorical columns with mode or 'UNKNOWN'
    cat_cols = ['full_name', 'username', 'email', 'city', 'employer']
    for col in cat_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna('UNKNOWN')
            
    # 3. Text Cleaning (.str.strip().str.upper())
    print("[INFO] Cleaning text columns...")
    text_columns = ['full_name', 'username', 'email', 'city', 'employer']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            
    # 4. 9-Rule Schema Validation Check (Demonstrating [FAIL] -> [PASS])
    print("\n--- 4. SCHEMA VALIDATION ---")
    def validate_schema(data, label):
        failures = []
        if data.empty: failures.append("Dataset is empty")
        if 'borrower_id' not in data.columns: failures.append("Missing borrower_id")
        if 'annual_income' not in data.columns: failures.append("Missing annual_income")
        if 'credit_score' not in data.columns: failures.append("Missing credit_score")
        if 'is_default' not in data.columns: failures.append("Missing is_default")
        if (data['annual_income'] < 0).any(): failures.append("Negative annual_income detected")
        if not data['credit_score'].between(300, 900).all(): failures.append("Credit score out of valid 300-900 range")
        if not data['is_default'].isin([0, 1]).all(): failures.append("Invalid is_default class label")
        if data.duplicated(subset=['borrower_id']).any(): failures.append("Duplicate borrower_ids found")
        
        if failures:
            print(f"[{label}] VALIDATION FAILED: {failures}")
            return False
        else:
            print(f"[{label}] VALIDATION PASSED successfully across all 9 rules.")
            return True

    # Intentionally inject a flaw in a test copy to show [FAIL]
    df_faulty = df.copy()
    df_faulty.loc[0, 'credit_score'] = 999
    validate_schema(df_faulty, "FAIL_DEMO")
    
    # Validate cleaned primary dataset
    is_valid = validate_schema(df, "PASS_DEMO")
    if not is_valid:
        raise ValueError("Data failed schema validation rules!")

    # 5. Outlier Capping via IQR and Z-score using .clip()
    print("[INFO] Capping outliers using IQR and Z-score methods...")
    for col in ['annual_income', 'credit_score']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Z-score clipping alternative
        mean_val = df[col].mean()
        std_val = df[col].std()
        z_lower = mean_val - 3 * std_val
        z_upper = mean_val + 3 * std_val
        
        # Combine bounds and clip
        final_lower = max(lower_bound, z_lower)
        final_upper = min(upper_bound, z_upper)
        df[col] = df[col].clip(lower=final_lower, upper=final_upper)

    # 6. Save Processed Dataset & Generate Sweetviz Report
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    processed_path = "data/processed/borrowers_cleaned.csv"
    df.to_csv(processed_path, index=False)
    print(f"[SUCCESS] Cleaned dataset saved to {processed_path}. Shape: {df.shape}")
    
    print("[INFO] Generating Sweetviz HTML exploratory profile report...")
    report = sv.analyze(df)
    report_path = "reports/sweetviz_report.html"
    report.show_html(report_path, open_browser=False)
    print(f"[SUCCESS] Sweetviz report generated at {report_path}")

if __name__ == "__main__":
    run_cleaning_pipeline()
