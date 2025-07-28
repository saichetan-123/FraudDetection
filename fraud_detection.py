import pandas as pd

# Load the dataset
df = pd.read_csv("creditcard.csv")

# Show first 5 rows
print("First 5 rows:\n", df.head())

# Shape of dataset
print("\nDataset shape:", df.shape)

# Count of normal vs fraud transactions
print("\nClass distribution:\n", df['Class'].value_counts())