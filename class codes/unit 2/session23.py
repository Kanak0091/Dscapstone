"""
Session 23: Unit 2 Capstone Lab, Interactive Data Visualization & Dashboard
Completes the DashboardBuilder scaffold with all 6 requirements:
  1 Explore  2 Analyze  3 Visualize (3+ charts)  4 Layout (important chart top-left)
  5 Story (titles state findings)  6 Interactivity (Plotly)
Install once:  pip install pandas plotly
Run:           python session23_dashboard_capstone.py
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class DashboardBuilder:
    COLORS = {"Delhi": "#1f77b4", "Mumbai": "#ff7f0e", "Pune": "#2ca02c"}

    def __init__(self, df):
        self.df = df

    # ---------- Requirement 1: Explore ----------
    def explore(self):
        print("Shape:", self.df.shape)
        print("\nInfo:")
        self.df.info()
        print("\nDescribe:\n", self.df.describe())
        print("\nCity counts:\n", self.df["City"].value_counts())

    # ---------- Requirement 2: Analyze ----------
    def analyze(self):
        num = self.df.select_dtypes("number")
        print("\nCorrelation matrix:\n", num.corr())
        print("\nAttendance-Marks correlation:",
              round(self.df["Attendance"].corr(self.df["Marks"]), 3))
        print("\nGroupby City:\n", self.df.groupby("City")[["Attendance", "Marks"]].mean())
        print("\nFinding: students with higher attendance score higher marks.")

    # ---------- Requirements 3-6: Visualize + Layout + Story + Interactivity ----------
    def visualize(self):
        city_avg = self.df.groupby("City", as_index=False)["Marks"].mean()

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"colspan": 2}, None], [{}, {}]],      # important chart top (full width)
            subplot_titles=(
                "Marks Rise With Attendance",              # most important -> top-left
                "Most Students Score Above 60",
                "Pune Students Lead on Average Marks",
            ),
            vertical_spacing=0.18,
        )

        # Scatter (Plotly: hover, zoom, legend toggle)
        for city, g in self.df.groupby("City"):
            fig.add_trace(
                go.Scatter(x=g["Attendance"], y=g["Marks"], mode="markers", name=city,
                           text=g["Name"], hovertemplate="%{text}<br>Attendance %{x}%<br>Marks %{y}",
                           marker=dict(size=12, color=self.COLORS[city])),
                row=1, col=1)

        # Histogram
        fig.add_trace(go.Histogram(x=self.df["Marks"], marker_color="#1f77b4",
                                   name="Marks", showlegend=False), row=2, col=1)

        # Bar
        fig.add_trace(go.Bar(x=city_avg["City"], y=city_avg["Marks"],
                             marker_color=[self.COLORS[c] for c in city_avg["City"]],
                             name="Avg Marks", showlegend=False), row=2, col=2)

        fig.update_xaxes(title_text="Attendance (%)", row=1, col=1)
        fig.update_yaxes(title_text="Marks (out of 100)", row=1, col=1)
        fig.update_xaxes(title_text="Marks (out of 100)", row=2, col=1)
        fig.update_yaxes(title_text="Students (count)", row=2, col=1)
        fig.update_yaxes(title_text="Average Marks", range=[0, 100], row=2, col=2)  # bar starts at 0

        fig.update_layout(
            title="Student Performance Dashboard",
            template="plotly_white", height=750,
        )
        fig.show()
        fig.write_html("dashboard_YourName_v1.html")       # exportable/shareable chart
        print("Saved dashboard_YourName_v1.html")
        return fig

    def run(self):
        self.explore()
        self.analyze()
        return self.visualize()


if __name__ == "__main__":
    df = pd.DataFrame({
        "Name":        ["Aarav", "Priya", "Rohan", "Sneha", "Karan", "Anjali", "Vikram", "Meera"],
        "City":        ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai", "Pune", "Delhi", "Mumbai"],
        "Attendance":  [92, 85, 70, 95, 60, 78, 55, 88],
        "Marks":       [88, 81, 64, 92, 52, 74, 45, 85],
        "Study_Hours": [6, 5, 3, 7, 2, 4, 2, 6],
    })
    DashboardBuilder(df).run()

    # One-paragraph reflection (write your own): what story does your dashboard tell?