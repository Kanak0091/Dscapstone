import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def run_eda_pipeline():
    print("[INFO] Loading cleaned processed dataset...")
    processed_path = "data/processed/borrowers_cleaned.csv"
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed data not found at {processed_path}. Run clean_validate.py first.")
        
    df = pd.read_csv(processed_path)
    os.makedirs("reports", exist_ok=True)
    
    # 1. Statistical Summaries & Groupby Analysis
    print("--- 1. STATISTICAL SUMMARIES ---")
    print(df[['annual_income', 'credit_score', 'late_count']].describe())
    
    print("\n--- 2. GROUPBY ANALYSIS (Default Rate by City) ---")
    city_summary = df.groupby('city').agg(
        total_borrowers=('borrower_id', 'count'),
        default_rate=('is_default', 'mean'),
        avg_income=('annual_income', 'mean')
    ).reset_index()
    print(city_summary)
    
    # 2. Generate Static Seaborn Charts (PNG)
    print("[INFO] Generating static Seaborn charts...")
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(8, 5))
    sns.histplot(df['annual_income'], kde=True, color='teal')
    plt.title("Distribution of Annual Income")
    plt.xlabel("Annual Income (INR)")
    plt.ylabel("Borrower Count")
    plt.tight_layout()
    plt.savefig("reports/income_distribution.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='is_default', y='credit_score', data=df, hue='is_default', palette='Set2', legend=False)
    plt.title("Credit Score vs Loan Default Status")
    plt.xlabel("Is Default (0 = Non-Default, 1 = Default)")
    plt.ylabel("Credit Score")
    plt.tight_layout()
    plt.savefig("reports/credit_vs_default.png", dpi=300)
    plt.close()
    print("[SUCCESS] Static PNG charts saved in reports/.")

    # 3. Build 4-Panel Interactive Plotly Dashboard
    print("[INFO] Building 4-panel interactive Plotly dashboard...")
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "1. Credit Score vs. Annual Income (Colored by Default)",
            "2. Average Annual Income by City",
            "3. Late Payment Count Distribution",
            "4. Credit Score Distribution by Default Status"
        )
    )
    
    # Panel 1: Scatter Plot (Income vs Credit Score)
    fig.add_trace(
        go.Scatter(
            x=df['annual_income'],
            y=df['credit_score'],
            mode='markers',
            marker=dict(size=10, color=df['is_default'], colorscale='Viridis', showscale=True),
            text=df['full_name'],
            name="Borrowers"
        ),
        row=1, col=1
    )
    
    # Panel 2: Bar Chart (Avg Income by City)
    fig.add_trace(
        go.Bar(
            x=city_summary['city'],
            y=city_summary['avg_income'],
            marker_color='indianred',
            name="Avg Income"
        ),
        row=1, col=2
    )
    
    # Panel 3: Histogram (Late Payment Count)
    fig.add_trace(
        go.Histogram(
            x=df['late_count'],
            marker_color='royalblue',
            name="Late Count"
        ),
        row=2, col=1
    )
    
    # Panel 4: Box Plot (Credit Score by Default)
    fig.add_trace(
        go.Box(
            x=df['is_default'].astype(str),
            y=df['credit_score'],
            marker_color='mediumseagreen',
            name="Credit Score Box"
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        width=1200,
        title_text="FinTech Loan Default & Credit Risk Analytics Dashboard (AML001)",
        showlegend=False
    )
    
    dashboard_path = "reports/eda_dashboard.html"
    fig.write_html(dashboard_path)
    print(f"[SUCCESS] Interactive Plotly dashboard saved to {dashboard_path}")

if __name__ == "__main__":
    run_eda_pipeline()
