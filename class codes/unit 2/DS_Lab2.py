import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
 
# ---------- Dataset ----------
df = pd.DataFrame({
    "Name": ["Asha", "Ravi", "Meera", "Kabir", "Sneha", "Arjun", "Isha", "Rohan"],
    "City": ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai", "Pune", "Delhi", "Mumbai"],
    "Attendance": [92, 65, 88, 70, 95, 58, 85, 72],
    "Marks": [88, 55, 80, 62, 92, 45, 78, 60],
    "Pass": ["Yes", "Yes", "Yes", "Yes", "Yes", "No", "Yes", "Yes"]
})
 
print(df.shape)
df.info()
print(df.describe())
print(df["City"].value_counts())

# ---------- Q1: Histogram, Boxplot, Scatter, Correlation ----------
plt.hist(df["Marks"], bins=5, color="teal", edgecolor="black")
plt.title("Distribution of Marks")
plt.xlabel("Marks")
plt.ylabel("Number of Students")
plt.show()
 
plt.boxplot(df["Marks"])
plt.title("Boxplot of Marks")
plt.show()
 
plt.scatter(df["Attendance"], df["Marks"])
plt.title("Attendance vs Marks")
plt.xlabel("Attendance")
plt.ylabel("Marks")
plt.show()
 
print(df["Attendance"].corr(df["Marks"]))
print(df[["Attendance", "Marks"]].corr())
 
sns.heatmap(df[["Attendance", "Marks"]].corr(), annot=True)
plt.title("Correlation Plot")
plt.show()

# ---------- Q2: Interactive Plotly dashboard ----------
fig1 = px.histogram(df, x="Marks", title="Distribution of Marks")
fig2 = px.scatter(df, x="Attendance", y="Marks", color="City",
                  hover_data=["Name"], title="Attendance vs Marks")
city_avg = df.groupby("City")["Marks"].mean().reset_index()
fig3 = px.bar(city_avg, x="City", y="Marks", title="Average Marks per City")
 
fig1.show()
fig2.show()
fig3.show()

# ---------- Q3: Time series and categorical ----------
ts = pd.DataFrame({
    "Month": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "Visitors": [1200, 1100, 1350, 1500, 1450, 1300, 1250, 1600, 1900, 2100, 2600, 3000]
})
ts = ts.sort_values("Month")
ts["Smoothed"] = ts["Visitors"].rolling(window=3).mean()
 
px.line(ts, x="Month", y="Visitors", title="Monthly Visitors").show()
px.line(ts, x="Month", y=["Visitors", "Smoothed"],
        title="Visitors with 3-Month Rolling Average").show()
 
print(df["City"].value_counts())
print(pd.crosstab(df["City"], df["Pass"]))
print(df.groupby(["City", "Pass"])["Marks"].mean())

# ---------- Q4: Storytelling ----------
fig = px.scatter(df, x="Attendance", y="Marks", color="City",
                 title="Students With Higher Attendance Get Higher Marks")
fig.show()
print("Correlation:", df["Attendance"].corr(df["Marks"]))
print("Takeaway: Better attendance goes with better marks.")

# ---------- Q5: Misleading vs correct chart ----------
sec = pd.DataFrame({"Section": ["A", "B"], "AvgMarks": [72, 75]})
 
bad = px.bar(sec, x="Section", y="AvgMarks", title="Section B Is Much Better")
bad.update_yaxes(range=[70, 76])
bad.show()
 
good = px.bar(sec, x="Section", y="AvgMarks",
              title="Sections A and B Have Similar Marks")
good.update_yaxes(range=[0, 100])
good.show()

# ---------- Q6: Compare two datasets ----------
df2 = pd.DataFrame({
    "Name": ["Maya", "Jay", "Lina", "Omar", "Zoya", "Hari", "Tina", "Nikhil"],
    "City": ["Delhi", "Mumbai", "Pune", "Delhi", "Mumbai", "Pune", "Delhi", "Mumbai"],
    "Attendance": [80, 78, 85, 74, 82, 70, 88, 76],
    "Marks": [68, 64, 72, 60, 70, 58, 76, 62],
    "Pass": ["Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"]
})
df["Batch"] = "Batch 1"
df2["Batch"] = "Batch 2"
both = pd.concat([df, df2])
 
print(both.groupby("Batch")["Marks"].mean())
print(both.groupby("Batch")["Marks"].std())
 
sns.boxplot(x="Batch", y="Marks", data=both)
plt.title("Marks: Batch 1 vs Batch 2")
plt.show()
 
px.scatter(both, x="Attendance", y="Marks", color="Batch",
           title="Attendance vs Marks by Batch").show()

# ---------- Q7: Report ----------
print(df["Marks"].mean(), df["Marks"].median(), df["Marks"].std())
print(df.groupby("City")["Marks"].mean())

