
import os
import json
from datetime import datetime
import requests
import pandas as pd
from sqlalchemy import create_engine, text

def fetch_borrower_profiles():
    print("[INFO] Fetching borrower profiles from public API...")
    # Using JSONPlaceholder users API to simulate borrower profiles for the FinTech pipeline
    url = "https://jsonplaceholder.typicode.com/users"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    raw_data = response.json()
    
    # Normalize JSON into a pandas DataFrame
    df = pd.json_normalize(raw_data)
    
    # Select and rename columns to match credit risk / borrower profile schema
    df = df[['id', 'name', 'username', 'email', 'address.city', 'company.name']]
    df.columns = ['borrower_id', 'full_name', 'username', 'email', 'city', 'employer']
    
    # Add simulated FinTech credit risk columns for AML001
    import numpy as np
    np.random.seed(42)
    df['annual_income'] = np.random.randint(30000, 150000, size=len(df))
    df['credit_score'] = np.random.randint(580, 850, size=len(df))
    df['is_default'] = np.random.choice([0, 1], size=len(df), p=[0.82, 0.18]) # ~18% minority class
    
    print(f"[SUCCESS] Fetched {len(df)} borrower profile records.")
    return df

def generate_sqlite_repayments():
    print("[INFO] Generating SQLite loan repayment database...")
    db_path = "data/raw/loans.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    engine = create_engine(f"sqlite:///{db_path}")
    
    # Create sample repayment history table
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE repayment_history (
                repayment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                borrower_id INTEGER,
                month_index INTEGER,
                payment_status TEXT,
                amount_paid REAL
            );
        """))
        
        # Insert mock repayment rows for 10 borrowers across 6 months
        import random
        random.seed(42)
        statuses = ['ON_TIME', 'LATE', 'DEFAULT']
        weights = [0.75, 0.20, 0.05]
        
        rows = []
        for b_id in range(1, 11):
            for m in range(1, 7):
                status = random.choices(statuses, weights=weights)[0]
                amt = round(random.uniform(500, 2500), 2)
                rows.append({"borrower_id": b_id, "month_index": m, "payment_status": status, "amount_paid": amt})
        
        conn.execute(
            text("INSERT INTO repayment_history (borrower_id, month_index, payment_status, amount_paid) VALUES (:borrower_id, :month_index, :payment_status, :amount_paid)"),
            rows
        )
        
    # Query back repayment summary via SQLAlchemy
    query = text("SELECT borrower_id, COUNT(*) as total_payments, SUM(CASE WHEN payment_status = 'LATE' THEN 1 ELSE 0 END) as late_count FROM repayment_history GROUP BY borrower_id")
    repay_df = pd.read_sql(query, con=engine)
    print(f"[SUCCESS] Queried {len(repay_df)} repayment summary records from SQLite.")
    return repay_df

def main():
    os.makedirs("data/raw", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Ingest REST API data
    df_borrowers = fetch_borrower_profiles()
    
    # 2. Ingest SQLite data
    df_repayments = generate_sqlite_repayments()
    
    # 3. Merge datasets on borrower_id
    df_merged = pd.merge(df_borrowers, df_repayments, on="borrower_id", how="left")
    df_merged['late_count'] = df_merged['late_count'].fillna(0)
    df_merged['total_payments'] = df_merged['total_payments'].fillna(6)
    
    # 4. Save dated raw CSV snapshot
    raw_csv_path = f"data/raw/borrowers_raw_{timestamp}.csv"
    df_merged.to_csv(raw_csv_path, index=False)
    
    # Also save a standard latest raw copy for downstream scripts
    df_merged.to_csv("data/raw/borrowers_raw_latest.csv", index=False)
    
    # 5. Save metadata.json audit trail
    metadata = {
        "student_name": "Kanak Chaudhary",
        "roll_no": "2601060001",
        "domain": "FinTech Loan Default & Credit Risk Analytics Pipeline",
        "timestamp": timestamp,
        "sources": [
            {"type": "REST API", "endpoint": "https://jsonplaceholder.typicode.com/users", "records": len(df_borrowers)},
            {"type": "SQLite Database", "db_file": "data/raw/loans.db", "table": "repayment_history"}
        ],
        "output_file": raw_csv_path,
        "total_rows": len(df_merged),
        "total_columns": len(df_merged.columns)
    }
    
    meta_path = f"data/raw/metadata_{timestamp}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    with open("data/raw/metadata_latest.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"[COMPLETE] Ingestion finished successfully. Snapshot saved to {raw_csv_path}")

if __name__ == "__main__":
    main()
