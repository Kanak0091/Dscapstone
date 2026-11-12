# %% Setup
import re
import requests
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)


# %% The DataToolkit class
class DataToolkit:
    # Reusable pipeline: fetch -> load -> inspect -> clean -> validate -> outliers -> transform

    # Columns that are names / places -> Title Case ; emails / websites -> lowercase
    TITLE_COLS = ["name", "address.street", "address.suite", "address.city", "company.name"]
    LOWER_COLS = ["email", "website"]
    # Columns the API sends as text but that are really numbers
    NUMERIC_TEXT_COLS = ["address.geo.lat", "address.geo.lng"]

    def __init__(self, url):
        self.url = url
        self.df = None
        # Keeps track of what we fixed, for the final summary
        self.report = {
            "rows_loaded": 0,
            "blank_to_missing": 0,
            "missing_filled": 0,
            "duplicates_removed": 0,
            "text_columns_standardised": 0,
            "columns_converted_to_numeric": 0,
            "validation_failures": 0,
            "outliers_flagged": 0,
        }

    # ---------- Requirement 1 & 2: Fetch & Load ----------
    def fetch_and_load(self):
        response = requests.get(self.url, timeout=15)
        print("Status code:", response.status_code)
        if response.status_code != 200:
            # Stop early: never continue with a failed API call
            raise RuntimeError(f"API call failed with status {response.status_code}")
        records = response.json()                 # list of dicts (nested)
        self.df = pd.json_normalize(records)      # flattens address.street, company.name, ...
        self.report["rows_loaded"] = len(self.df)
        print("Shape:", self.df.shape)
        return self.df

    # ---------- Requirement 3: Inspect ----------
    def inspect(self):
        df = self.df
        print("Shape:", df.shape)
        print("\n--- info() ---")
        df.info()
        print("\n--- describe() ---")
        print(df.describe(include="all").T)
        print("\n--- Missing values per column ---")
        print(df.isnull().sum())
        print("\nDuplicate rows:", df.duplicated().sum())

    # ---------- Requirement 4: Clean ----------
    def clean(self):
        df = self.df.copy()

        # 1) Blank strings are really missing values
        blanks = int(df.isin(["", " "]).sum().sum())
        df = df.replace(r"^\s*$", np.nan, regex=True)
        self.report["blank_to_missing"] = blanks

        # 2) Columns stored as text but really numeric -> convert
        for col in self.NUMERIC_TEXT_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                self.report["columns_converted_to_numeric"] += 1

        # 3) Missing values: numeric -> median, text -> "Unknown"
        missing_before = int(df.isnull().sum().sum())
        num_cols = df.select_dtypes(include="number").columns
        text_cols = [c for c in df.columns if c not in num_cols]
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
        for col in text_cols:
            df[col] = df[col].fillna("Unknown")
        self.report["missing_filled"] = missing_before

        # 4) Duplicates
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        self.report["duplicates_removed"] = before - len(df)

        # 5) Standardise text (assign the result back! .str methods return a NEW Series)
        for col in text_cols:
            df[col] = df[col].astype(str).str.strip()
        for col in self.TITLE_COLS:
            if col in df.columns:
                df[col] = df[col].str.title()
                self.report["text_columns_standardised"] += 1
        for col in self.LOWER_COLS:
            if col in df.columns:
                df[col] = df[col].str.lower()
                self.report["text_columns_standardised"] += 1

        self.df = df
        # Checkpoint from the lab sheet
        print("Missing values left :", int(df.isnull().sum().sum()))
        print("Duplicates left     :", int(df.duplicated().sum()))
        return df

    # ---------- Requirement 5: Validate ----------
    # Schema: column -> expected type, whether required, and a rule (True = row is valid)
    SCHEMA = {
        "id":              {"type": "int",   "required": True, "desc": "positive, unique integer",
                            "rule": lambda s: (s > 0) & ~s.duplicated()},
        "name":            {"type": "str",   "required": True, "desc": "non-empty text",
                            "rule": lambda s: s.str.len() > 0},
        "username":        {"type": "str",   "required": True, "desc": "non-empty, no spaces",
                            "rule": lambda s: (s.str.len() > 0) & ~s.str.contains(" ")},
        "email":           {"type": "str",   "required": True, "desc": "contains '@' and a domain (name@site.tld)",
                            "rule": lambda s: s.str.match(r"^[\w\.\-+]+@[\w\-]+\.[\w\.\-]+$", na=False)},
        "phone":           {"type": "str",   "required": True, "desc": "at least 7 digits",
                            "rule": lambda s: s.str.count(r"\d") >= 7},
        "website":         {"type": "str",   "required": True, "desc": "contains a '.' and no spaces",
                            "rule": lambda s: s.str.contains(r"\.", regex=True) & ~s.str.contains(" ")},
        "address.city":    {"type": "str",   "required": True, "desc": "non-empty text",
                            "rule": lambda s: s.str.len() > 0},
        "address.zipcode": {"type": "str",   "required": True, "desc": "non-empty text",
                            "rule": lambda s: s.str.len() > 0},
        "address.geo.lat": {"type": "float", "required": True, "desc": "between -90 and 90",
                            "rule": lambda s: s.between(-90, 90)},
        "address.geo.lng": {"type": "float", "required": True, "desc": "between -180 and 180",
                            "rule": lambda s: s.between(-180, 180)},
        "company.name":    {"type": "str",   "required": True, "desc": "non-empty text",
                            "rule": lambda s: s.str.len() > 0},
    }

    def validate(self):
        type_checks = {
            "int": pd.api.types.is_integer_dtype,
            "float": pd.api.types.is_float_dtype,
            "str": pd.api.types.is_string_dtype,
        }
        rows = []
        for col, spec in self.SCHEMA.items():
            if col not in self.df.columns:
                rows.append({"column": col, "present": False, "dtype_ok": False, "bad_rows": None,
                             "status": "FAIL (missing)" if spec["required"] else "skip", "rule": spec["desc"]})
                continue
            s = self.df[col]
            dtype_ok = bool(type_checks[spec["type"]](s))
            rule_ok = spec["rule"](s) if dtype_ok else pd.Series(False, index=s.index)
            bad = int((~rule_ok).sum())
            rows.append({"column": col, "present": True, "dtype_ok": dtype_ok, "actual_dtype": str(s.dtype),
                         "bad_rows": bad, "status": "PASS" if (dtype_ok and bad == 0) else "FAIL",
                         "rule": spec["desc"]})
        result = pd.DataFrame(rows)
        self.report["validation_failures"] = int(result["status"].str.startswith("FAIL").sum())
        print(result.to_string(index=False))
        print(f"\nValidation: {int((result['status'] == 'PASS').sum())} of {len(result)} checks passed.")
        return result

    # ---------- Requirement 6: Outliers ----------
    def outliers(self, col, method="iqr", threshold=None):
        s = self.df[col]
        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (s < low) | (s > high)
            print(f"[IQR] {col}: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f} -> fences [{low:.2f}, {high:.2f}]")
        elif method == "zscore":
            threshold = 3 if threshold is None else threshold
            z = (s - s.mean()) / s.std()          # std() uses ddof=1 (sample std)
            mask = z.abs() > threshold
            print(f"[Z-score] {col}: threshold=|z|>{threshold}, largest |z| in data = {z.abs().max():.2f}")
        else:
            raise ValueError("method must be 'iqr' or 'zscore'")
        flagged = int(mask.sum())
        self.report["outliers_flagged"] += flagged
        print(f"          -> {flagged} outlier(s) flagged out of {len(s)} rows")
        return self.df[mask]

    # ---------- Requirement 7: Transform & Summarize ----------
    def transform(self):
        df = self.df.copy()
        # Derived column
        df["email_domain"] = df["email"].str.split("@").str[1]
        # Rename the clumsy flattened names
        df = df.rename(columns={
            "address.street": "street", "address.suite": "suite", "address.city": "city",
            "address.zipcode": "zipcode", "address.geo.lat": "latitude", "address.geo.lng": "longitude",
            "company.name": "company_name", "company.catchPhrase": "company_catchphrase",
            "company.bs": "company_bs",
        })
        # Sort alphabetically by name so lookups are easy
        df = df.sort_values("name").reset_index(drop=True)
        self.df = df
        return df

    def summary(self):
        r = self.report
        print("=" * 55)
        print("FINAL SUMMARY")
        print("=" * 55)
        print(f"Rows x Columns (final)      : {self.df.shape[0]} x {self.df.shape[1]}")
        print(f"Rows loaded from API        : {r['rows_loaded']}")
        print(f"Blank strings -> missing    : {r['blank_to_missing']}")
        print(f"Missing values filled       : {r['missing_filled']}")
        print(f"Duplicate rows removed      : {r['duplicates_removed']}")
        print(f"Text columns standardised   : {r['text_columns_standardised']}")
        print(f"Columns converted to number : {r['columns_converted_to_numeric']}")
        print(f"Schema checks failed        : {r['validation_failures']}")
        print(f"Outliers flagged            : {r['outliers_flagged']}")
        print("New column added            : email_domain")
        print("=" * 55)


# %% Requirements 1 & 2: Fetch & Load
URL = "https://jsonplaceholder.typicode.com/users"
tk = DataToolkit(URL)
df = tk.fetch_and_load()
print(df.head())

# %% Requirement 3: Inspect
tk.inspect()

# %% Requirement 4: Clean
tk.clean()
print(tk.df.head())

# %% Requirement 5: Validate (print real dtypes first, don't guess)
print(tk.df.dtypes)
print()
validation_result = tk.validate()

# %% Requirement 6: Outliers (IQR and Z-score)
# With only 10 rows these methods are unstable. For Z-score, the largest possible
# |z| with n rows is (n-1)/sqrt(n) = 2.85 for n=10, so a cut-off of 3 can NEVER flag anything.
tk.df["username_length"] = tk.df["username"].str.len()

print("--- IQR method ---")
out_len_iqr = tk.outliers("username_length", method="iqr")
out_lat_iqr = tk.outliers("address.geo.lat", method="iqr")

print("\n--- Z-score method ---")
out_len_z = tk.outliers("username_length", method="zscore", threshold=3)

n = len(tk.df)
print(f"\nMax possible |z| with n={n}: {(n - 1) / np.sqrt(n):.2f}  (cut-off is 3, so it can never be exceeded)")

out_len_z2 = tk.outliers("username_length", method="zscore", threshold=2)

print("\nRows flagged by IQR on username_length:")
print(out_len_iqr[["username", "username_length"]] if len(out_len_iqr) else "None")

# %% Requirement 7: Transform & Summarize
clean_df = tk.transform()
print(clean_df.head())

print("\nUsers per email domain:")
print(clean_df["email_domain"].value_counts())

print("\nUsers with usernames longer than 8 characters:")
print(clean_df.loc[clean_df["username_length"] > 8, ["name", "username", "username_length"]])

tk.summary()

# %% Reflection (write your own version in your report / notebook)
# Hardest part: the outlier step. With 10 rows I had to think about whether IQR or Z-score
# makes sense at all. Also nested JSON: pd.DataFrame() left the address dict inside one cell
# until I used pd.json_normalize().