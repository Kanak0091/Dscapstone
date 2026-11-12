# FinTech Loan Default & Credit Risk Analytics Pipeline (AML001)

- **Student Name:** Kanak Chaudhary
- **Roll No:** 2601060001
- **Domain:** FinTech Loan Default & Credit Risk Analytics Pipeline (predicting binary `is_default` ~18% minority class)
- **Environment:** VS Code using WSL (Ubuntu) on Windows

---

## Project Structure
- `data/raw/`: Timestamps raw snapshots and SQLite database (`loans.db`).
- `data/processed/`: Cleaned and validated dataset (`borrowers_cleaned.csv`).
- `notebooks/`: Jupyter notebook workspace.
- `src/`:
  - `ingest.py`: Multi-source REST API & SQLite ingestion with metadata logging.
  - `clean_validate.py`: Data cleaning, 9-rule schema validation, and Sweetviz profiling.
  - `eda_dashboard.py`: Statistical summaries, Seaborn static plots, and interactive Plotly dashboard.
  - `train_pipeline.py`: Feature engineering, SMOTE balancing, ML model comparison, and inference testing.
- `reports/`: HTML reports, experiment metrics JSON, and PNG charts.
- `models/`: Versioned model bundles (`.pkl`).
