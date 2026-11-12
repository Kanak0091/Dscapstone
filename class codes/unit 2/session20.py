"""
Session 20: Dashboard Design & Reporting Best Practices
Install once:  pip install pandas plotly
Run:           python session20_dashboard_design.py
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.DataFrame({
    "Name":       ["Aarav", "Priya", "Rohan", "Sneha", "Karan", "Anjali", "Vikram", "Meera"],
    "City":       ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai", "Pune", "Delhi", "Mumbai"],
    "Attendance": [92, 85, 70, 95, 60, 78, 55, 88],
    "Marks":      [88, 81, 64, 92, 52, 74, 45, 85],
})

# ----- Code Example 1: Multi-chart layout with make_subplots -----
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Marks Distribution", "Attendance vs. Marks"),
)
fig.show()   # empty layout, the building block

# ----- Fill it: most important chart top-left, one consistent palette -----
colors = {"Delhi": "#1f77b4", "Mumbai": "#ff7f0e", "Pune": "#2ca02c"}

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Most Students Score Above 60", "Marks Rise With Attendance"),
)
fig.add_trace(go.Histogram(x=df["Marks"], marker_color="#1f77b4", name="Marks"), row=1, col=1)
for city, g in df.groupby("City"):
    fig.add_trace(
        go.Scatter(x=g["Attendance"], y=g["Marks"], mode="markers", name=city,
                   text=g["Name"], marker=dict(size=11, color=colors[city])),
        row=1, col=2,
    )
fig.update_xaxes(title_text="Marks (out of 100)", row=1, col=1)
fig.update_yaxes(title_text="Students (count)", row=1, col=1)
fig.update_xaxes(title_text="Attendance (%)", row=1, col=2)
fig.update_yaxes(title_text="Marks (out of 100)", row=1, col=2)

# Reporting best practice: title + source + sample size + units
fig.update_layout(
    title="Student Performance Dashboard",
    template="plotly_white",
    annotations=list(fig.layout.annotations) + [
        dict(text="Source: class records | n = 8 students | Marks out of 100",
             x=0, y=-0.2, xref="paper", yref="paper", showarrow=False)
    ],
)
fig.show()

# ----- Code Example 2: Title that states the finding -----
fig = px.scatter(df, x="Attendance", y="Marks", title="Marks Rise With Attendance")
fig.show()

# ----- 5-second rule: show it to a classmate for 5 seconds, then hide it -----