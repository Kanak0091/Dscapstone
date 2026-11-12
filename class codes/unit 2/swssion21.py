"""
Session 21: Data Bias, Leakage & Ethical Visualization
(Slides have no code, so these are small demos of each idea.)
Install once:  pip install pandas numpy plotly
Run:           python session21_bias_leakage_ethics.py
"""
import numpy as np
import pandas as pd
import plotly.express as px

# =========================================================
# 1) DATA LEAKAGE: predicting whether a student will FAIL
# =========================================================
df = pd.DataFrame({
    "Attendance":        [92, 85, 70, 95, 60, 78, 55, 88],
    "Marks_So_Far":      [88, 81, 64, 92, 52, 74, 45, 85],
    "final_result_flag": [0, 0, 0, 0, 1, 0, 1, 0],        # 1 = failed (known only AFTER result)
})
print("Correlation of each feature with the target:")
print(df.corr()["final_result_flag"].drop("final_result_flag"))

features_ok   = ["Attendance", "Marks_So_Far"]   # known BEFORE predicting -> OK
features_leak = ["final_result_flag"]            # known only AFTER result  -> LEAK
print("\nOK features  :", features_ok)
print("LEAK feature :", features_leak, "<- drop it from X!")
print('Ask: "Would I actually know this value BEFORE making the prediction?"')

# =========================================================
# 2) ETHICAL TRAP 1: Truncated axis
# =========================================================
data = pd.DataFrame({"Team": ["A", "B"], "Score": [95, 97]})   # only a ~2% difference

bad = px.bar(data, x="Team", y="Score", title="MISLEADING: axis starts at 90")
bad.update_yaxes(range=[90, 98])
bad.show()

good = px.bar(data, x="Team", y="Score", title="HONEST: axis starts at 0")
good.update_yaxes(range=[0, 100])
good.show()

# =========================================================
# 3) ETHICAL TRAP 2: Misleading scale (radius vs area)
# =========================================================
r1, r2 = 1, 2
print(f"\nRadius doubled ({r1} -> {r2}); area grows {(np.pi*r2**2)/(np.pi*r1**2):.0f}x, not 2x")

bubble = pd.DataFrame({"Item": ["X", "Y"], "Value": [10, 20], "y": [0, 0]})
px.scatter(bubble, x="Item", y="y", size="Value", title="Bubble (area-based): hard to compare").show()
px.bar(bubble, x="Item", y="Value", title="Bar (length-based): honest comparison").show()

# =========================================================
# 4) ETHICAL TRAP 3: Cherry-picking time ranges
# =========================================================
ts = pd.DataFrame({
    "Month": pd.date_range("2025-01-01", periods=8, freq="MS"),
    "Sales": [100, 98, 95, 60, 62, 70, 85, 90],
})
px.line(ts.iloc[3:], x="Month", y="Sales", title="CHERRY-PICKED: 'Sales up 40%!'").show()
px.line(ts, x="Month", y="Sales", title="FULL PICTURE: sales still below Jan").show()

# =========================================================
# 5) DATA BIAS: sampling bias quick demo
# =========================================================
rng = np.random.default_rng(0)
population = pd.DataFrame({"Marks": rng.normal(60, 15, 1000)})
biased_sample = population.sort_values("Marks", ascending=False).head(100)  # only high scorers
print(f"\nPopulation mean: {population['Marks'].mean():.1f}")
print(f"Biased sample mean (only toppers answered): {biased_sample['Marks'].mean():.1f}")