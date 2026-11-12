import pandas as pd

data = {
    "RollNo": [101, 102, 103, 104, 105],
    "Name": ["Aarav", "Diya", "Rohan", "Sneha", "Kabir"],
    "Age": [18, 19, 18, 19, 18],
    "Course": ["BCA", "BCA", "BCA", "BCA", "BCA"],
    "Marks": [78, 85, 62, 91, 70],
}

df = pd.DataFrame(data)
print(df)
