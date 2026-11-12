"""
Session 22: Power BI / Tableau Concepts
(Slides have no code. Here is the Python equivalent of each BI concept.)
Install once:  pip install pandas plotly
Run:           python session22_powerbi_tableau_concepts.py
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ----- Connect to data:  BI "Get Data" button  ==  pd.read_csv() -----
df = pd.DataFrame({
    "Name":       ["Aarav", "Priya", "Rohan", "Sneha", "Karan", "Anjali", "Vikram", "Meera"],
    "City":       ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai", "Pune", "Delhi", "Mumbai"],
    "Attendance": [92, 85, 70, 95, 60, 78, 55, 88],
    "Marks":      [88, 81, 64, 92, 52, 74, 45, 85],
})
# df = pd.read_csv("students.csv")     # same as BI "Get Data"

# ----- Drag-and-drop visual:  drag column to axis  ==  px.bar(df, x=..., y=...) -----
avg = df.groupby("City", as_index=False)["Marks"].mean()
px.bar(avg, x="City", y="Marks", title="Average Marks by City").show()

# ----- Slicer: viewer filters the dashboard themselves (dropdown) -----
cities = df["City"].unique().tolist()
fig = go.Figure()
for c in cities:
    g = df[df["City"] == c]
    fig.add_trace(go.Scatter(x=g["Attendance"], y=g["Marks"], mode="markers",
                             name=c, text=g["Name"], marker=dict(size=12)))

buttons = [dict(label="All", method="update", args=[{"visible": [True] * len(cities)}])]
for i, c in enumerate(cities):
    vis = [j == i for j in range(len(cities))]
    buttons.append(dict(label=c, method="update", args=[{"visible": vis}]))

fig.update_layout(
    title="Slicer Demo: pick a City from the dropdown",
    xaxis_title="Attendance (%)", yaxis_title="Marks (out of 100)",
    updatemenus=[dict(buttons=buttons, direction="down", x=0.02, y=1.15)],
)
fig.show()

# ----- Publish & share:  BI cloud publish  ==  sharing a Plotly HTML file -----
fig.write_html("slicer_dashboard.html")
print("Saved slicer_dashboard.html (open or share this file)")