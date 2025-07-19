import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
df = pd.read_excel('aa.xlsx')

# Clean column names
df.columns = df.columns.str.strip().str.replace('\n', '').str.replace('\r', '')

# Check if 'Variety' column exists
if 'Variety' not in df.columns:
    raise KeyError("The 'Variety' column is missing from the DataFrame.")

# Convert 'Year-Week' to datetime format and extract features
df['Year-Week'] = df['Year-Week'].apply(lambda x: f"{x[:4]}-W{x[5:]}-1")
df['Year-Week'] = pd.to_datetime(df['Year-Week'], format='%Y-W%W-%w')
df['Year'] = df['Year-Week'].dt.year\
df['Week'] = df['Year-Week'].dt.isocalendar().week

# Prepare features and target (including temporal features like Year and Week)
X = df[['Year', 'Week', 'arrivals(tons)', 'sales(tones)', 'T max (o C)', 'T min (o C)', 'Rainfall (mm)', 'Area  (\'000 hectares)', 'Production(\'000 tonnes)', 'Variety']]
y = df['Teja_price(tons)']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and train models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Save the trained models as pickle files
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open('gradient_boosting_model.pkl', 'wb') as f:
    pickle.dump(gb_model, f)

# Function to load the model based on user selection
def load_model(variety, model_name, year, week):
    # Load the appropriate model
    if model_name == 'RandomForest':
        with open('random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
    elif model_name == 'GradientBoosting':
        with open('gradient_boosting_model.pkl', 'rb') as f:
            model = pickle.load(f)
    else:
        raise ValueError("Model not recognized. Choose 'RandomForest' or 'GradientBoosting'.")

    # Filter the data for the selected year and week
    filtered_data = df[(df['Year'] == year) & (df['Week'] == week)]

    # Prepare the features for prediction
    X_new = filtered_data[['Year', 'Week', 'arrivals(tons)', 'sales(tones)', 'T max (o C)', 'T min (o C)', 'Rainfall (mm)', 'Area  (\'000 hectares)', 'Production(\'000 tonnes)']]

    # Predict using the loaded model
    predictions = model.predict(X_new)
    return predictions

# Example usage
variety = 'Teja'
model_name = 'RandomForest'
year = 2023
week = 10
predictions = load_model(variety, model_name, year, week)
print(f'Predictions for {variety} using {model_name} model for Year {year}, Week {week}: {predictions}')
