import pandas as pd
import requests , numpy as np , sweetviz as sv

#1. Collect data from a REST API (e.g., JSONPlaceholder Users API)
api_url = "https://jsonplaceholder.typicode.com/users"
response = requests.get(api_url)
api_data = response.json()
df_api = pd.DataFrame(api_data)

#2. Load data from a CSV file (e.g., local sales/transactions dataset)
#Replace 'dataset.csv' with your actual file path
df_csv = pd.read_csv("student-dataset.csv")

#Display raw data
print("--- API Raw Data Preview ---")
print(df_api.head())

print("\n--- CSV Raw Data Preview ---")
print(df_csv.head())



# Select one primary dataset to demonstrate (e.g., df_csv or df_api)
df = df_csv.copy()

# 1. Shape of the dataset (rows, columns)
print(f"Dataset Shape: {df.shape}")

# 2. Data types and non-null counts
print("\n--- Data Types & Info ---")
df.info()

# 3. Summary statistics for numerical columns
print("\n--- Summary Statistics (Numerical) ---")
print(df.describe())

# 4. Summary statistics for categorical/object columns
print("\n--- Summary Statistics (Categorical) ---")
print(df.describe)


#--- BEFORE TREATMENT ---
print("Missing values before treatment:")
print(df.isnull().sum())
print(f"\nDuplicate rows before treatment: {df.duplicated().sum()}")

# --- TREATMENT ---
# 1. Remove duplicate records
df_cleaned = df.drop_duplicates().copy()

# 2. Drop completely empty column ('ethnic.group' is entirely null in this dataset)
df_cleaned = df_cleaned.drop(columns=["ethnic.group"])

# 3. Handle any other specific missing columns if needed (e.g., math.grade or city)
df_cleaned["math.grade"] = df_cleaned["math.grade"].fillna(
    df_cleaned["math.grade"].median()
)

# --- AFTER TREATMENT ---
print("\n--- AFTER TREATMENT ---")
print("Missing values after treatment:")
print(df_cleaned.isnull().sum())
print(f"Duplicate rows after treatment: {df_cleaned.duplicated().sum()}")


#Define approved categories and required columns
allowed_genders = ["F", "M", "other"]
required_cols = ["id", "name", "age", "gender", "english.grade", "math.grade"]

def validate_student_dataset(df_cleaned):
    # Dictionary holding all our validation rules
    checks = {
        "1. No missing required fields": df_cleaned[required_cols].isnull().sum().sum() == 0,
        "2. Age within range (18-30)": df_cleaned["age"].between(18, 30).all(),
        "3. English grade in range (0-4.0)": df_cleaned["english.grade"].between(0.0, 4.0).all(),
        "4. Math grade in range (0-4.0)": df_cleaned["math.grade"].between(0.0, 4.0).all(),
        "5. Gender categories are valid": df_cleaned["gender"].isin(allowed_genders).all(),
        "6. No duplicate rows": df_cleaned.duplicated().sum() == 0,
        "7. Student IDs are unique": df_cleaned["id"].is_unique
    }
    return checks

# Run the validation function
validation_results = validate_student_dataset(df_cleaned)

# Print pass/fail status for each check
for check_name, passed in validation_results.items():
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {check_name}")


#Select a single numeric column to analyze for outliers
col = "english.grade"

print("=== 1. IQR METHOD OUTLIERS ===")
# Find Q1 (25th percentile) and Q3 (75th percentile) using pandas quantile()
Q1 = df_cleaned[col].quantile(0.25)
Q3 = df_cleaned[col].quantile(0.75)

# Calculate the Interquartile Range (spread of the middle 50% of data)
IQR = Q3 - Q1

# Define the lower and upper bounds using the standard 1.5 multiplier rule
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Create a boolean filter to flag rows that fall outside the acceptable bounds
is_outlier_iqr = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)

# Print the student ID, name, and grade for the flagged IQR outliers
print(df_cleaned[is_outlier_iqr][["id", "name", col]])


print("\n=== 2. Z-SCORE METHOD OUTLIERS ===")
# Calculate the mean (average) and standard deviation (spread) for the column
mean = df_cleaned[col].mean()
std = df_cleaned[col].std()

# Compute the Z-score for each value to find how many standard deviations it is from the mean
z_scores = (df_cleaned[col] - mean) / std
df_cleaned["z_score"] = z_scores

# Flag values where the absolute Z-score is greater than 3 (meaning it's unusually far from normal)
is_outlier_z = df_cleaned["z_score"].abs() > 3

# Print the student ID, name, grade, and computed Z-score for the flagged Z-score outliers
print(df_cleaned[is_outlier_z][["id", "name", col, "z_score"]])


print(lower_bound)
print(upper_bound)

sweet_report = sv.analyze(df_cleaned)
sweet_report.show_html("sweetviz_report.html")

# 3. Export to Excel for ML Workflows
output_file = "student_dataset_cleaned.xlsx"
df_cleaned.to_excel(output_file, index=False)