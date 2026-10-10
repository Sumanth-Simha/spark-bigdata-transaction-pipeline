import pandas as pd

# Load the full parquet dataset
df = pd.read_parquet('data/parquet/creditcard')

# Sample 90 non-fraud and 10 fraud cases so there's an interesting mix for the threshold
legit = df[df['Class'] == 0].sample(90, random_state=42)
fraud = df[df['Class'] == 1].sample(min(10, len(df[df['Class'] == 1])), random_state=42)

# Combine and shuffle
test_df = pd.concat([legit, fraud]).sample(frac=1, random_state=42).reset_index(drop=True)

# Drop 'Class' to simulate real-world blind inference data
test_df = test_df.drop(columns=['Class'])

test_df.to_csv('test_transactions.csv', index=False)
print("Created test_transactions.csv with 100 rows.")
