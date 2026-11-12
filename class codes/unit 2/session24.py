"""
Session 24: Revision & Problem Solving, Units 1 & 2
Code for the 4 practice questions + the 4 common pitfalls.
Install once:  pip install pandas numpy plotly requests
Run:           python session24_revision.py
"""
import numpy as np
import pandas as pd
import plotly.express as px

# =========================================================
# Practice Q1: mean = 60 but median = 78  -> what does it suggest?
# =========================================================
marks = pd.Series([95, 90, 85, 80, 78, 76, 15, 10, 5, 66])
print("Q1 Mean  :", marks.mean())
print("   Median:", marks.median())
print("   Mean < Median -> a few very low outliers drag the mean down (left/negative skew)")
print("   Skewness:", round(marks.skew(), 2), "(negative = left skew)")

# =========================================================
# Practice Q2: 99% test accuracy, fails in production -> check DATA LEAKAGE
# =========================================================
leaky = pd.DataFrame({
    "Income":      [30, 50, 20, 80, 40],
    "Age":         [25, 40, 22, 50, 33],
    "default_flag_after": [1, 0, 1, 0, 1],     # known only AFTER the outcome
    "Default":     [1, 0, 1, 0, 1],            # target
})
print("\nQ2 Correlation with target:")
print(leaky.corr()["Default"].drop("Default"))   # perfect 1.0 = leakage red flag
X = leaky.drop(columns=["Default", "default_flag_after"])   # remove the leaking column
print("   Features kept:", list(X.columns))

# =========================================================
# Practice Q3: three numeric columns together -> MULTIVARIATE analysis
# =========================================================
df = pd.DataFrame({
    "Attendance":  [92, 85, 70, 95, 60, 78, 55, 88],
    "Marks":       [88, 81, 64, 92, 52, 74, 45, 85],
    "Study_Hours": [6, 5, 3, 7, 2, 4, 2, 6],
    "City":        ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai", "Pune", "Delhi", "Mumbai"],
})
print("\nQ3 Correlation matrix:")
print(df[["Attendance", "Marks", "Study_Hours"]].corr().round(2))
print("\n   Multi-column groupby:")
print(df.groupby("City")[["Attendance", "Marks", "Study_Hours"]].mean())
px.imshow(df[["Attendance", "Marks", "Study_Hours"]].corr(), text_auto=".2f",
          title="Multivariate Analysis: Correlation Matrix").show()

# =========================================================
# Practice Q4: Y-axis starts at 80 -> TRUNCATED AXIS
# =========================================================
d = pd.DataFrame({"Team": ["A", "B"], "Score": [82, 84]})
fig = px.bar(d, x="Team", y="Score", title="Truncated axis (misleading)")
fig.update_yaxes(range=[80, 85])
fig.show()
fig = px.bar(d, x="Team", y="Score", title="Axis from zero (honest)")
fig.update_yaxes(range=[0, 100])
fig.show()

# =========================================================
# Common Pitfalls
# =========================================================
# 1) dropna() removes rows; fillna() replaces values
dirty = pd.DataFrame({"A": [1, np.nan, 3], "B": [4, 5, np.nan]})
print("\nPitfall 1: dropna ->\n", dirty.dropna())
print("fillna(mean) ->\n", dirty.fillna(dirty.mean()))

# 2) sort_values() BEFORE plotting a time-series
ts = pd.DataFrame({"Month": pd.to_datetime(["2025-03-01", "2025-01-01", "2025-02-01"]),
                   "Visitors": [300, 100, 200]})
ts = ts.sort_values("Month")
print("\nPitfall 2: sorted time-series ->\n", ts)

# 3) Correlation is NOT causation
ice = pd.DataFrame({"IceCream": [10, 20, 30, 40], "Drownings": [1, 2, 3, 4]})
print("\nPitfall 3: correlation =", ice["IceCream"].corr(ice["Drownings"]),
      "(both rise in summer; one doesn't cause the other)")

# 4) Check status_code before trusting an API response
import requests
try:
    r = requests.get("https://jsonplaceholder.typicode.com/posts/1", timeout=10)
    if r.status_code == 200:
        print("\nPitfall 4: OK, got data:", r.json()["title"][:40], "...")
    else:
        print("\nPitfall 4: request failed with status", r.status_code)
except requests.RequestException as e:
    print("\nPitfall 4: request error (check internet):", e)