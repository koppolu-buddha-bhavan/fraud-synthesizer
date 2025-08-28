import pandas as pd
from sklearn.preprocessing import StandardScaler


# Load and preprocess the dataset, and export DataFrames for use in other modules
DATASET_PATH = 'Creditcard_dataset.csv'
df = pd.read_csv(DATASET_PATH)
df = df.dropna()
if 'Time' in df.columns:
    df = df.drop(columns=['Time'])
scaler = StandardScaler()
if 'Amount' in df.columns:
    df['Amount'] = scaler.fit_transform(df[['Amount']])

# Split into features and labels
X = df.drop(columns=['Class'])
y = df['Class']

# Split into fraud and genuine
df_fraud = df[df['Class'] == 1]
df_genuine = df[df['Class'] == 0]
