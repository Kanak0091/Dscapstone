import pandas as pd
import requests

api_url = "https://jsonplaceholder.typicode.com/users"
response = requests.get(api_url)
df_api = pd.DataFrame(response.json())

df_csv = pd.read_csv("student-dataset.csv")

print("--- API Raw Data Preview ---")
print(df_api.head())

print("\n--- CSV Raw Data Preview ---")
print(df_csv.head())

print(df_csv.columns)

df = df_csv.copy()

print(f"Dataset Shape: {df.shape}")

print("\n--- Data Types & Info ---")

df.info()

print("\n--- Summary Statistics (Numerical) ---")
print(df.describe())
print("\n--- Summary Statistics (Categorical) ---")

print(df.describe(include="object"))

print("Missing values before treatment:")
print(df.isnull().sum())
print(f"\nDuplicate rows before treatment: {df.duplicated().sum()}")

df_cleaned = df.drop_duplicates().copy()

df_cleaned = df_cleaned.drop(columns=["ethnic.group"])

df_cleaned["math.grade"] = df_cleaned["math.grade"].fillna(df_cleaned["math.grade"].median())

print("\n--- AFTER TREATMENT ---")
print("Missing values after treatment:")
print(df_cleaned.isnull().sum())
print(f"Duplicate rows after treatment: {df_cleaned.duplicated().sum()}")

allowed_genders = ["F", "M", "other"]

required_cols = ["id", "name", "age", "gender", "english.grade", "math.grade"]

def validate_student_dataset(df_cleaned):
    checks = {
        "1. No missing required fields": df_cleaned[required_cols].isnull().sum().sum() == 0,
        "2. Age within range (18-30)": df_cleaned["age"].between(18, 30).all(),
        "3. English grade in range (0-4.0)": df_cleaned["english.grade"].between(0.0, 4.0).all(),
        "4. Math grade in range (0-4.0)": df_cleaned["math.grade"].between(0.0, 4.0).all(),
        "5. Gender categories are valid": df_cleaned["gender"].isin(allowed_genders).all(),
        "6. No duplicate rows": df_cleaned.duplicated().sum() == 0,
        "7. Student IDs are unique": df_cleaned["id"].is_unique,
    }
    return checks

validation_results = validate_student_dataset(df_cleaned)

for check_name, passed in validation_results.items():
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {check_name}")
   
   
    col = "english.grade"

print("=== 1. IQR METHOD OUTLIERS ===")

Q1 = df_cleaned[col].quantile(0.25)
Q3 = df_cleaned[col].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

is_outlier_iqr = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)

print(df_cleaned[is_outlier_iqr][["id", "name", col]])

print("\n=== 2. Z-SCORE METHOD OUTLIERS ===")

mean = df_cleaned[col].mean()
std = df_cleaned[col].std()
df_cleaned["z_score"] = (df_cleaned[col] - mean) / std
is_outlier_z = df_cleaned["z_score"].abs() > 3

print(df_cleaned[is_outlier_z][["id", "name", col, "z_score"]])

print(lower_bound)
print(upper_bound)

import sweetviz as sv

sweet_report = sv.analyze(df_cleaned)
sweet_report.show_html("sweetviz_report.html")