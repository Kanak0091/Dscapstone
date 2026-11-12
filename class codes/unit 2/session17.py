import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

"""
Session 17: Interactive Dashboards using Plotly
Install once:  pip install pandas plotly
Run:           python session17_plotly_dashboards.py
Each fig.show() opens an interactive chart in your browser.
"""
import pandas as pd
import plotly.express as px

# ----- DataFrame -----
df = pd.DataFrame({
    "Name":        ["Aarav", "Priya", "Rohan", "Sneha", "Karan", "Anjali", "Vikram", "Meera"],
    "City":        ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai", "Pune", "Delhi", "Mumbai"],
    "Attendance":  [92, 85, 70, 95, 60, 78, 55, 88],
    "Marks":       [88, 81, 64, 92, 52, 74, 45, 85],
    "Study_Hours": [6, 5, 3, 7, 2, 4, 2, 6],
})
print(df)

# ----- Code Example 1: Plotly histogram -----
fig = px.histogram(df, x="Marks")
fig.show()

# ----- Code Example 2: Interactive scatter plot (hover shows student name) -----
fig = px.scatter(df, x="Attendance", y="Marks", color="City", hover_data=["Name"])
fig.show()

# ----- Code Example 3: Grouped bar chart (average Marks per City) -----
city_avg = df.groupby("City")["Marks"].mean().reset_index()
print(city_avg)
fig = px.bar(city_avg, x="City", y="Marks")
fig.show()

# ----- Code Example 4: Multiple charts together -----
fig1 = px.histogram(df, x="Marks")
fig2 = px.scatter(df, x="Attendance", y="Marks", color="City")
fig1.show()
fig2.show()

# ----- In-class activity -----
# 1. Scatter of Attendance vs Marks coloured by City  -> done above
# 2. Hover over 3 points and read the tooltips
# 3. Click a city in the legend to hide/show it
# 4. Bar chart of average Marks per City using groupby() + px.bar() -> done above