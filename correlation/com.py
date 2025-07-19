import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Step 1: Load the data
# Assuming the data is in an Excel file named 'aa.xlsx' and located in the parent folder
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

# Step 4: Create a 'Date' column by combining 'Year' and 'Month' for time-based analysis
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))

# Step 5: Visualize the relationship between each factor and Price (e.g., 'Teja_price')

# 1. Price vs Arrivals (tons)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='arrivals', y='Teja_price')
plt.title('Teja Price vs Arrivals')
plt.xlabel('Arrivals (tons)')
plt.ylabel('Teja Price')
plt.show()

# 2. Price vs Sales (tons)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='sales', y='Teja_price')
plt.title('Teja Price vs Sales')
plt.xlabel('Sales (tons)')
plt.ylabel('Teja Price')
plt.show()

# 3. Price vs Temperature (T max, °C)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='T_max', y='Teja_price')
plt.title('Teja Price vs Max Temperature (°C)')
plt.xlabel('Max Temperature (°C)')
plt.ylabel('Teja Price')
plt.show()

# 4. Price vs Temperature (T min, °C)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='T_min', y='Teja_price')
plt.title('Teja Price vs Min Temperature (°C)')
plt.xlabel('Min Temperature (°C)')
plt.ylabel('Teja Price')
plt.show()

# 5. Price vs Rainfall (mm)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Rainfall', y='Teja_price')
plt.title('Teja Price vs Rainfall (mm)')
plt.xlabel('Rainfall (mm)')
plt.ylabel('Teja Price')
plt.show()

# 6. Price vs Area ('000 hectares)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Area', y='Teja_price')
plt.title('Teja Price vs Area (\'000 hectares)')
plt.xlabel('Area (\'000 hectares)')
plt.ylabel('Teja Price')
plt.show()

# 7. Price vs Production ('000 tonnes)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Production', y='Teja_price')
plt.title('Teja Price vs Production (\'000 tonnes)')
plt.xlabel('Production (\'000 tonnes)')
plt.ylabel('Teja Price')
plt.show()

df['Date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)
# Step 6: Check for correlation between variables, including 'Date_ordinal'
correlation_matrix = df[[ 'arrivals', 'sales', 'T_max', 'T_min', 'Rainfall', 'Area', 'Production', 'Teja_price']].corr()
# Plot heatmap of correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1, linecolor='black')
plt.title('Correlation Matrix')
plt.show()

df = df.dropna(subset=['Date', 'Teja_price'])
# Step 6: Visualize Price Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Date', y='Teja_price')
plt.title('Teja Price Over Time')
plt.xlabel('Date')
plt.ylabel('Teja Price')
plt.xticks(rotation=45)
plt.show()

# Step 8: Pairplot to Show Relationships Between All Variables
sns.pairplot(df[['arrivals', 'sales', 'T_max', 'T_min', 'Rainfall', 'Area', 'Production', 'Teja_price']])
plt.show()
