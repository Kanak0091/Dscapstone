"""
Session 19: Business-Oriented Data Storytelling
(The slides have no code, so this shows the concepts in code:
 a BAD chart vs a GOOD chart, and the one-sentence takeaway rule.)
Install once:  pip install pandas numpy plotly
Run:           python session19_data_storytelling.py
"""
import numpy as np
import pandas as pd
import plotly.express as px

# ----- DataFrame: monthly sales for 8 products -----
rng = np.random.default_rng(42)
months = pd.date_range("2025-01-01", periods=12, freq="MS")
sales = pd.DataFrame({f"Product {c}": rng.integers(50, 150, 12) for c in "ABCDEFGH"})
sales["Month"] = months
sales["Product A"] = np.linspace(60, 160, 12).round()        # Product A is the real story
long_df = sales.melt(id_vars="Month", var_name="Product", value_name="Sales")

# ----- BAD chart: 8 colours, no title, no takeaway (chartjunk / too much data) -----
bad = px.line(long_df, x="Month", y="Sales", color="Product")
bad.update_layout(title=None)
bad.show()

# ----- GOOD chart: one idea, one highlight, title states the finding -----
long_df["Highlight"] = np.where(long_df["Product"] == "Product A", "Product A", "Others")
good = px.line(
    long_df, x="Month", y="Sales", line_group="Product", color="Highlight",
    color_discrete_map={"Product A": "#d62728", "Others": "#cccccc"},
    title="Product A Sales Grew Steadily All Year While Others Stayed Flat",
    labels={"Sales": "Units sold (count)"},
)
good.show()

# ----- One-sentence takeaway rule -----
print('This chart shows that Product A outsold the others and grew every month.')

# ----- Right chart for the message -----
# "Sales grew steadily"        -> Line chart
# "Product A outsold others"   -> Bar chart
# "Attendance predicts Marks"  -> Scatter plot