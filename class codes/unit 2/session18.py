"""
Session 18: Time-Series Visualization & Trend Analysis (CampusMart case study)
Install once:  pip install pandas plotly
Run:           python session18_time_series.py
"""
import pandas as pd
import plotly.express as px

# ----- DataFrame: 12 months of visitors (deliberately NOT in date order) -----
data = {
    "Month": pd.to_datetime([
        "2025-03-01", "2025-01-01", "2025-07-01", "2025-12-01",
        "2025-05-01", "2025-02-01", "2025-09-01", "2025-11-01",
        "2025-04-01", "2025-10-01", "2025-06-01", "2025-08-01",
    ]),
    "Visitors": [5200, 4500, 6600, 5100, 5900, 4800, 7100, 7600, 5500, 7400, 6200, 6900],
}
df = pd.DataFrame(data)

print("before sorting")
print(df)

# ----- Without sorting: the line zigzags meaninglessly -----
px.line(df, x="Month", y="Visitors", title="UNSORTED (wrong)").show()

# ----- Code Example 1: Sort, then plot -----
df = df.sort_values("Month")
print("\nafter sorting")
print(df)
px.line(df, x="Month", y="Visitors", title="Sorted by Month").show()

# ----- Code Example 2: 3-month rolling average -----
df["Smoothed"] = df["Visitors"].rolling(window=3).mean()
print("\nwith 3-month rolling average")
print(df)

# ----- Plot both lines together -----
fig = px.line(
    df, x="Month", y=["Visitors", "Smoothed"],
    title="Traffic Grows All Year, With a Predictable December Dip",
    labels={"value": "Visitors (count)", "variable": "Series"},
)
fig.show()

# ----- Reading the trend -----
# Trend       : overall growth
# Seasonality : yearly December dip (exam season)
# Insight     : plan marketing AROUND the dip, don't panic at it