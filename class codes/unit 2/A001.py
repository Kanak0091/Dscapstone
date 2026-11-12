import pandas as pd
import numpy as np

# =====================================================
# Create a separate DataFrame (sample student data)
# It has missing values, duplicates and mixed gender labels
# =====================================================
data = {
    "id":      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10],
    "name":    ["Aman", "Riya", "Sam", "Neha", "Ravi", "Pooja",
                "Karan", "Simran", "Arjun", "Meera", "Meera"],
    "gender":  ["M", "Female", "m", "F", "Male", "F",
                "M", "f", "Male", "F", "F"],
    "age":     [18, 19, 20, None, 19, 21, 20, 18, None, 19, 19],
    "maths":   [78, 85, None, 90, 66, 72, None, 88, 91, 60, 60],
    "english": [82, 79, 88, 75, None, 69, 84, 90, 77, 73, 73],
}

df = pd.DataFrame(data)

# Save it as a CSV file so we can practise loading a CSV (Q3)
df.to_csv("marks_data.csv", index=False)
print("marks_data.csv created in the current folder\n")


# =====================================================
# Q3. Load the CSV, show first 5 rows, info() and describe()
# =====================================================
print("=========== Q3 ===========")

# Load the CSV file into a DataFrame
df = pd.read_csv("marks_data.csv")

# Show the first 5 rows
print("--- First 5 rows ---")
print(df.head())

# Show column names, data types and non-null counts
print("\n--- Info ---")
df.info()

# Show summary statistics of numeric columns
print("\n--- Describe ---")
print(df.describe())


# =====================================================
# Q6. Count missing values per column and fill them
# =====================================================
print("\n=========== Q6 ===========")

# Count missing values in each column
print("--- Missing values per column (before) ---")
print(df.isnull().sum())

# Work on a copy so the original data stays safe
df_filled = df.copy()

# Fill missing values in 'age' with the median
df_filled["age"] = df_filled["age"].fillna(df_filled["age"].median())

# Fill missing values in 'maths' with the mean
df_filled["maths"] = df_filled["maths"].fillna(df_filled["maths"].mean())

# Fill missing values in 'english' with the mean
df_filled["english"] = df_filled["english"].fillna(df_filled["english"].mean())

# Check again to confirm no missing values are left
print("\n--- Missing values per column (after) ---")
print(df_filled.isnull().sum())

print("\n--- Data after filling ---")
print(df_filled)


# =====================================================
# Extra (Q4): remove duplicates and fix gender labels
# =====================================================
print("\n=========== Q4 (cleaning) ===========")

# Remove duplicate rows
df_clean = df_filled.drop_duplicates()
print("Rows after removing duplicates:", len(df_clean))

# Make gender labels consistent (M, Male, m -> M ; F, Female, f -> F)
df_clean = df_clean.copy()
df_clean["gender"] = df_clean["gender"].str.strip().str.lower().map(
    {"m": "M", "male": "M", "f": "F", "female": "F"}
)

print(df_clean)
print("\nGender counts:")
print(df_clean["gender"].value_counts())