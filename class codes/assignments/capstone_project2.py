# %% Setup
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", None)


# %% Dataset: synthetic 60 students (fixed seed, so results are repeatable)
rng = np.random.default_rng(42)
n = 60

branch = rng.choice(["AI & DS", "Cyber Security", "Cloud Computing", "Software Dev"],
                    size=n, p=[0.35, 0.20, 0.20, 0.25])
study_hours = np.clip(rng.normal(3.5, 1.2, n), 0.5, 7).round(1)
attendance = np.clip(rng.normal(80, 10, n), 50, 100).round(0)
sleep_hours = np.clip(rng.normal(6.8, 1.0, n), 4, 9).round(1)

months = pd.date_range("2026-08-01", "2026-12-01", freq="MS")
test_month = months[rng.integers(0, len(months), n)]

# Exam score depends mostly on study hours, a bit on attendance and sleep, plus random noise
exam_score = (12 + 8 * study_hours + 0.3 * attendance + 1.5 * (sleep_hours - 6)
              + rng.normal(0, 6, n))
exam_score = np.clip(exam_score, 0, 100).round(1)

raw_df = pd.DataFrame({
    "student_id": [f"S{i:03d}" for i in range(1, n + 1)],
    "branch": branch,
    "study_hours": study_hours,
    "attendance_pct": attendance,
    "sleep_hours": sleep_hours,
    "exam_score": exam_score,
    "test_month": test_month,
})
# grade is DERIVED from exam_score (a leakage trap if used to "explain" exam_score)
raw_df["grade"] = pd.cut(raw_df["exam_score"], bins=[-1, 49.9, 59.9, 69.9, 100],
                         labels=["D", "C", "B", "A"])
print(raw_df.head())


# %% The DashboardBuilder class
class DashboardBuilder:
    NUM_COLS = ["study_hours", "attendance_pct", "sleep_hours", "exam_score"]

    def __init__(self, df):
        self.df = df.copy()
        self.findings = {}   # numbers computed in analyze(), reused in chart titles
        self.figs = {}       # individual Plotly figures built in visualize()

    # ---------- Requirement 1: Explore ----------
    def explore(self):
        df = self.df
        print("Shape:", df.shape)
        print("\n--- info() ---")
        df.info()
        print("\n--- describe() ---")
        print(df.describe().round(2))
        print("\n--- value_counts(): branch ---")
        print(df["branch"].value_counts())
        print("\n--- value_counts(): grade ---")
        print(df["grade"].value_counts())
        print("\nMissing values:", int(df.isnull().sum().sum()))

    # ---------- Requirement 2: Analyze ----------
    def analyze(self):
        df = self.df
        corr = df[self.NUM_COLS].corr()
        by_branch = (df.groupby("branch")["exam_score"]
                       .agg(students="count", mean_score="mean", median_score="median")
                       .round(1).sort_values("mean_score", ascending=False))
        monthly = df.groupby("test_month")["exam_score"].mean().round(1)

        slope, intercept = np.polyfit(df["study_hours"], df["exam_score"], 1)
        self.findings = {
            "corr": corr,
            "by_branch": by_branch,
            "monthly": monthly,
            "r_study": corr.loc["study_hours", "exam_score"],
            "r_attend": corr.loc["attendance_pct", "exam_score"],
            "r_sleep": corr.loc["sleep_hours", "exam_score"],
            "slope": slope,
            "intercept": intercept,
            "top_branch": by_branch.index[0],
            "branch_gap": by_branch["mean_score"].max() - by_branch["mean_score"].min(),
            "mean_score": df["exam_score"].mean(),
        }
        print("Correlation matrix:")
        print(corr.round(2))
        print("\nAverage exam score by branch (groupby):")
        print(by_branch)
        f = self.findings
        print(f"\nFINDING: study_hours and exam_score have correlation r = {f['r_study']:.2f}; "
              f"each extra hour of daily study goes with about {f['slope']:.1f} more marks.")

        # Static heatmap (Matplotlib / Seaborn). Close the window to continue.
        plt.figure(figsize=(5.5, 4))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", vmin=-1, vmax=1)
        plt.title("Study hours show the strongest link with exam score")
        plt.tight_layout()
        plt.show()

    # ---------- Ethics: bias & leakage ----------
    def check_ethics(self):
        df = self.df
        print("BIAS CHECK: is every group represented?")
        counts = df["branch"].value_counts()
        print(counts.to_string())
        print(f"Smallest group has {counts.min()} students ({counts.min() / len(df):.0%} of the data). "
              "Averages for small groups are noisy, so I show group sizes next to the averages.")

        print("\nLEAKAGE CHECK:")
        leak_corr = df[["exam_score"]].assign(grade_code=df["grade"].cat.codes).corr().iloc[0, 1]
        print(f"- 'grade' is built from exam_score (correlation with it: {leak_corr:.2f}). "
              "Using it to explain the score would be circular, so it is EXCLUDED from every chart.")
        print("- study_hours, attendance_pct and sleep_hours are recorded independently of the score, so they are safe to use.")
        print("\nHONESTY NOTES:")
        print("- Correlation is not causation: studying more may go with higher scores, but this data cannot prove it.")
        print("- The y-axis of the bar chart starts at 0 so differences are not exaggerated.")
        print("- The dataset is synthetic and small (60 rows), so results should not be generalised.")

    # ---------- Requirements 3, 5, 6: Visualize (finding titles + Plotly) ----------
    def visualize(self, show=True):
        df, f = self.df, self.findings
        figs = {}

        # (1) Scatter: study hours vs exam score (MOST IMPORTANT chart -> top-left)
        fig = go.Figure()
        for b in sorted(df["branch"].unique()):
            sub = df[df["branch"] == b]
            fig.add_trace(go.Scatter(x=sub["study_hours"], y=sub["exam_score"], mode="markers",
                                     name=b, marker=dict(size=8, opacity=0.8),
                                     hovertemplate="%{text}<br>Study: %{x} h/day<br>Score: %{y}<extra>" + b + "</extra>",
                                     text=sub["student_id"]))
        xs = np.array([df["study_hours"].min(), df["study_hours"].max()])
        fig.add_trace(go.Scatter(x=xs, y=f["slope"] * xs + f["intercept"], mode="lines",
                                 name="Trend line", line=dict(color="black", dash="dash")))
        fig.update_layout(title=f"Every extra study hour adds about {f['slope']:.0f} marks (r={f['r_study']:.2f})",
                          xaxis_title="Study hours per day", yaxis_title="Exam score (out of 100)")
        figs["scatter"] = fig

        # (2) Bar: average score by branch (y-axis starts at 0 = honest)
        bb = f["by_branch"]
        if f["branch_gap"] < 5:
            bar_title = f"Branch matters little: averages differ by only {f['branch_gap']:.1f} marks"
        else:
            bar_title = f"{f['top_branch']} leads, {f['branch_gap']:.1f} marks above the lowest branch"
        fig = go.Figure(go.Bar(x=bb.index, y=bb["mean_score"], marker_color="#4C78A8",
                               text=[f"{m:.1f}<br>(n={c})" for m, c in zip(bb["mean_score"], bb["students"])],
                               textposition="outside",
                               hovertemplate="%{x}<br>Average: %{y}<extra></extra>"))
        fig.update_layout(title=bar_title, xaxis_title="Branch", yaxis_title="Average exam score",
                          yaxis_range=[0, 100], showlegend=False)
        figs["bar"] = fig

        # (3) Histogram: distribution of exam scores
        fig = go.Figure(go.Histogram(x=df["exam_score"], nbinsx=10, marker_color="#54A24B"))
        share_pass = (df["exam_score"] >= 50).mean()
        fig.update_layout(title=f"{share_pass:.0%} of students score 50 or above (average {f['mean_score']:.0f})",
                          xaxis_title="Exam score", yaxis_title="Number of students", showlegend=False)
        figs["hist"] = fig

        # (4) Line: average score by test month (time-ordered bonus)
        m = f["monthly"]
        direction = "rose" if m.iloc[-1] > m.iloc[0] else "fell"
        fig = go.Figure(go.Scatter(x=m.index, y=m.values, mode="lines+markers", line=dict(color="#E45756"),
                                   hovertemplate="%{x|%b %Y}<br>Average: %{y}<extra></extra>"))
        fig.update_layout(title=f"Monthly average {direction} from {m.iloc[0]:.0f} ({m.index[0]:%b}) to {m.iloc[-1]:.0f} ({m.index[-1]:%b})",
                          xaxis_title="Test month", yaxis_title="Average exam score", showlegend=False,
                          xaxis_tickformat="%b %Y")
        figs["line"] = fig

        self.figs = figs
        if show:
            for fig in figs.values():
                fig.show()
        return figs

    # ---------- Requirement 4: Layout (5-second rule: key chart top-left) ----------
    def build_dashboard(self, html_path="student_dashboard.html"):
        f = self.findings
        order = [("scatter", 1, 1), ("bar", 1, 2), ("hist", 2, 1), ("line", 2, 2)]
        wrapped_titles = ["<br>".join(textwrap.wrap(self.figs[k].layout.title.text, 50)) for k, _, _ in order]

        dash = make_subplots(rows=2, cols=2, subplot_titles=wrapped_titles,
                             horizontal_spacing=0.10, vertical_spacing=0.20)
        for key, r, c in order:
            src = self.figs[key]
            for trace in src.data:
                dash.add_trace(trace, row=r, col=c)
            dash.update_xaxes(title_text=src.layout.xaxis.title.text, row=r, col=c)
            dash.update_yaxes(title_text=src.layout.yaxis.title.text, row=r, col=c)
            if key == "bar":
                dash.update_yaxes(range=[0, 100], row=r, col=c)
        dash.update_xaxes(tickformat="%b %Y", row=2, col=2)

        for ann in dash.layout.annotations:     # make subplot titles smaller
            ann.update(font=dict(size=12))
        dash.update_layout(
            title=dict(text=f"Student Performance Dashboard: study time is the strongest driver of exam scores (r = {f['r_study']:.2f})",
                       font=dict(size=17)),
            height=820, width=1100, margin=dict(t=120, b=60, l=60, r=40),
            legend=dict(orientation="h", y=-0.12),
            template="plotly_white",
        )
        dash.write_html(html_path, include_plotlyjs="cdn")   # shareable, interactive file
        print("Saved interactive dashboard to", html_path)
        return dash


# %% Requirement 1: Explore
db = DashboardBuilder(raw_df)
db.explore()

# %% Requirement 2: Analyze
db.analyze()

# %% Ethics checkpoint: bias & leakage
db.check_ethics()

# %% Requirements 3, 5 & 6: Visualize (4 interactive Plotly charts; they open in your browser)
figs = db.visualize()

# %% Requirement 4: Layout (combined dashboard)
dashboard = db.build_dashboard("student_dashboard.html")
dashboard.show()

# %% Export a shareable chart
db.figs["scatter"].write_html("scatter_study_vs_score.html", include_plotlyjs="cdn")
print("Saved scatter_study_vs_score.html")
print("Both HTML files are saved in the same folder as this script.")