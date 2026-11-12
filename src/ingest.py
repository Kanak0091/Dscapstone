import os
import json
from datetime import datetime
import requests
import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np

def fetch_borrower_profiles():
    print("[INFO] Fetching borrower profiles...")
    url = "https://jsonplaceholder.typicode.com/users"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    raw_data = response.json()
    
    df_base = pd.json_normalize(raw_data)
    df_base = df_base[['id', 'name', 'username', 'email', 'address.city', 'company.name']]
    df_base.columns = ['borrower_id', 'full_name', 'username', 'email', 'city', 'employer']
    
    # Expand to 100 synthetic borrower profiles to ensure robust ML train/test split and SMOTE balance
    np.random.seed(42)
    n_samples = 100
    
    expanded_data = []
    for i in range(1, n_samples + 1):
        base_row = df_base.sample(n=1, replace=True).iloc[0].to_dict()
        base_row['borrower_id'] = i
        base_row['annual_income'] = int(np.random.randint(30000, 150000))
        base_row['credit_score'] = int(np.random.randint(580, 850))
        base_row['is_default'] = int(np.random.choice([0, 1], p=[0.82, 0.18]))
        expanded_data.append(base_row)
        
    df = pd.DataFrame(expanded_data)
    print(f"[SUCCESS] Generated {len(df)} borrower profile records.")
    return df

def generate_sqlite_repayments():
    print("[INFO] Generating SQLite loan repayment database...")
    db_path = "data/raw/loans.db"
    if os.path.exists(db_path):
        os.path.exists(db_path) and os.remove(db_path)
        
    engine = create_engine(f"sqlite:///{db_path}")
    
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
        
        import random
        random.seed(42)
        statuses = ['ON_TIME', 'LATE', 'DEFAULT']
        weights = [0.75, 0.20, 0.05]
        
        rows = []
        for b_id in range(1, 101):
            for m in range(1, 7):
                status = random.choices(statuses, weights=weights)[0]
                amt = round(random.uniform(500, 2500), 2)
                rows.append({"borrower_id": b_id, "month_index": m, "payment_status": status, "amount_paid": amt})
        
        conn.execute(
            text("INSERT INTO repayment_history (borrower_id, month_index, payment_status, amount_paid) VALUES (:borrower_id, :month_index, :payment_status, :amount_paid)"),
            rows
        )
        
    query = text("SELECT borrower_id, COUNT(*) as total_payments, SUM(CASE WHEN payment_status = 'LATE' THEN 1 ELSE 0 END) as late_count FROM repayment_history GROUP BY borrower_id")
    repay_df = pd.read_sql(query, con=engine)
    print(f"[SUCCESS] Queried {len(repay_df)} repayment summary records from SQLite.")
    return repay_df

def main():
    os.makedirs("data/raw", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    df_borrowers = fetch_borrower_profiles()
    df_repayments = generate_sqlite_repayments()
    
    df_merged = pd.merge(df_borrowers, df_repayments, on="borrower_id", how="left")
    df_merged['late_count'] = df_merged['late_count'].fillna(0)
    df_merged['total_payments'] = df_merged['total_payments'].fillna(6)
    
    raw_csv_path = f"data/raw/borrowers_raw_{timestamp}.csv"
    df_merged.to_csv(raw_csv_path, index=False)
    df_merged.to_csv("data/raw/borrowers_raw_latest.csv", index=False)
    
    metadata = {
        "student_name": "Kanak Chaudhary",
        "roll_no": "2601060001",
        "domain": "FinTech Loan Default & Credit Risk Analytics Pipeline",
        "timestamp": timestamp,
        "sources": [
            {"type": "REST API + Synthetic Expansion", "records": len(df_borrowers)},
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
