import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the data
df = pd.read_excel('bb.xlsx')

# Step 2: Display first few rows of the data to inspect
print(df.head())

# Step 3: Data Preprocessing
# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Fill missing values with forward fill method (or you can handle it differently)
df = df.fillna(method='ffill')

# Check for duplicates and drop them if any
df = df.drop_duplicates()

# Step 4: Ensure 'Date' is in datetime format and create 'Date' column if necessary
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1), errors='coerce')

# Step 5: Ensure 'Teja_price' is numeric
df['Teja_price'] = pd.to_numeric(df['Teja_price'], errors='coerce')

# Remove rows where there is any missing data for the selected columns
df = df.dropna(subset=['arrivals', 'sales', 'T_max', 'T_min', 'Rainfall', 'Area', 'Production', 'Teja_price'])

# Step 6: Check dimensions and data types for selected columns
print(df[['arrivals', 'sales', 'T_max', 'T_min', 'Rainfall', 'Area', 'Production', 'Teja_price']].dtypes)
print(f"Number of rows after dropping missing values: {df.shape[0]}")

# Step 7: Pairplot to Show Relationships Between All Variables
sns.pairplot(df[['arrivals', 'sales', 'T_max', 'T_min', 'Rainfall', 'Area', 'Production', 'Teja_price']])
plt.show()
