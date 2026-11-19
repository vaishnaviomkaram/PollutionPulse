import pandas as pd

# Load the data
df = pd.read_csv("data/raw/city_day.csv")

# See the shape (rows, columns)
print("Shape:", df.shape)

# See first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# See all column names
print("\nColumns:")
print(df.columns.tolist())

# See how many cities
print("\nNumber of cities:", df["City"].nunique())
print("City names:", df["City"].unique())