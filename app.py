
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd 
import numpy as np  # Make sure to import numpy
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
import json  # Ensure this line is included
from prophet import Prophet  # Ensure you have Prophet installed
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import io
import base64
from prophet.diagnostics import cross_validation, performance_metrics
import warnings



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/forcastindex')
def forcastindex():
    return render_template('forcastindex.html')



@app.route('/forecasting24')
def forecasting24():
    # Load your historical data
    data = pd.read_excel('max.xlsx')

    # Clean column names by stripping any leading/trailing spaces
    data.columns = data.columns.str.strip()

    # Prepare 'Price' data for Prophet model 
    data['ds'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1))
    data = data.rename(columns={'Price': 'y'})  # Rename 'Price' column to 'y' for Prophet
    data = data.dropna(subset=['y', 'ds'])  # Drop rows with missing values in 'y' or 'ds'

    # Forecast Arrivals separately
    arrivals_data = data[['ds', 'Arrivals']].dropna()
    arrivals_data = arrivals_data.rename(columns={'Arrivals': 'y'})

    # Create Prophet model for Arrivals forecast
    arrivals_model = Prophet(seasonality_mode='multiplicative', changepoint_prior_scale=10)
    arrivals_model.add_seasonality(name='monthly', period=30.5, fourier_order=2)
    arrivals_model.fit(arrivals_data)

    # Make future dataframe for arrivals forecast
    arrivals_future = arrivals_model.make_future_dataframe(periods=12, freq='M')
    arrivals_forecast = arrivals_model.predict(arrivals_future)

    # Prepare 'Price' model
    model = Prophet(seasonality_mode='multiplicative', changepoint_prior_scale=10)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=2)

    # Rename forecasted arrivals to 'Arrivals' before merging
    forecasted_arrivals = arrivals_forecast[['ds', 'yhat']].rename(columns={'yhat': 'Arrivals'})

    # Merge forecasted_arrivals into the data
    data = data.merge(forecasted_arrivals[['ds', 'Arrivals']], on='ds', how='left')

    # After merging, ensure only the correct 'Arrivals' column is used
    data = data.drop(columns=['Arrivals_x'], errors='ignore')  # Remove 'Arrivals_x' if exists
    data = data.rename(columns={'Arrivals_y': 'Arrivals'})  # Rename 'Arrivals_y' to 'Arrivals'

    # Fit the price model with 'Arrivals' as a regressor
    model.add_regressor('Arrivals')
    model.fit(data[['ds', 'y', 'Arrivals']])

    # Make future dataframe for price forecast
    future = model.make_future_dataframe(periods=12, freq='M')
    future = future.merge(forecasted_arrivals[['ds', 'Arrivals']], on='ds', how='left')

    # Ensure 'Arrivals' column in future data
    future['Arrivals'] = future['Arrivals'].fillna(0)  # Fill missing arrivals with 0 if any

    # Forecast Price
    forecast = model.predict(future)

    # Ensure no negative values in forecasted results
    forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, 0))
    forecast['yhat_lower'] = forecast['yhat_lower'].apply(lambda x: max(x, 0))
    forecast['yhat_upper'] = forecast['yhat_upper'].apply(lambda x: max(x, 0))

    # Load actual future data for comparison (if available)
    actual_data = pd.read_excel('max.xlsx')  # Replace with your actual data file
    actual_data['ds'] = pd.to_datetime(actual_data[['Year', 'Month']].assign(DAY=1))
    # Merge forecast with actual data
    merged = forecast.merge(actual_data[['ds', 'Price']], on='ds', how='left')
    merged.rename(columns={'Price': 'actual'}, inplace=True)

    # Calculate accuracy metrics
    merged['error'] = merged['actual'] - merged['yhat']
    merged['abs_error'] = merged['error'].abs()
    merged['abs_percentage_error'] = (merged['abs_error'] / merged['actual']) * 100
    merged['accuracy'] = 100 - merged['abs_percentage_error']

    # Calculate overall average absolute percentage error
    overall_avg_accuracy = merged['accuracy'].mean() if not merged['accuracy'].isnull().all() else None
    merged.fillna(0, inplace=True)

    # Sort data by date in descending order
    merged_sorted = merged.sort_values(by='ds', ascending=False)

    # Prepare merged data for template
    merged_data = merged_sorted[['ds', 'actual', 'yhat', 'error', 'abs_error', 'abs_percentage_error', 'accuracy']].to_dict(orient='records')


    # Plot with Plotly
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Price vs Arrivals"), specs=[[{"secondary_y": True}]])
    
    # Trace for Historical Price (show only one line for actual)
    trace1 = go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='<b>Historical Price (Tons) </b>', line=dict(color='blue'))

    # Trace for Forecast Price
    trace2 = go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='<b>Forecast Price  (Tons) </b>', line=dict(color='green'))

    # Trace for Actual Arrivals (Bar for bags in Lakhs)
    trace3 = go.Bar(x=actual_data['ds'], y=actual_data['Arrivals'], name='<b>Actual Arrivals (Tons)</b>',
                    marker=dict(color='orange', opacity=0.6), yaxis='y2')

    # Trace for Forecasted Arrivals (Differently styled for next year)
    trace4 = go.Scatter(x=arrivals_forecast['ds'], y=arrivals_forecast['yhat'], mode='lines', name='<b>Forecasted Arrivals (Tons)</b>',
                        line=dict(color='red', dash='dash'), yaxis='y2')

    # Add the traces to the figure
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.add_trace(trace3)  # Actual Arrivals with secondary y-axis
    fig.add_trace(trace4)  # Forecasted Arrivals with secondary y-axis

    # Update layout
    fig.update_layout(
        title='',
        xaxis=dict(
            title='<b>Month</b>',
            tickfont=dict(size=14, family='Arial Black', color='black')
        ),
        yaxis=dict(
            title='<b>Price (per Tons)</b>',
            tickprefix='<b>₹</b>',  # Adds bold Indian Rupee symbol as prefix
            tickfont=dict(size=14, family='Arial Black', color='black')
        ),
        yaxis2=dict(
            title='<b>Arrival in Tons</b>',
           
            tickfont=dict(size=14, family='Arial Black', color='black')
        ),
        hovermode='x unified',
        height=600
    )

    # Convert the plot to JSON for rendering in the HTML
    graph_json = fig.to_json()

    # Pass the merged data and overall average accuracy percentage to the template
    return render_template('forecasting.html', graph_json=graph_json, merged_data=merged_data, overall_avg_accuracy=overall_avg_accuracy)



@app.route('/forecasting12')
def forecasting12():
    # Load your data
    data = pd.read_excel('bb.xlsx')

    # Ensure required columns exist
    required_columns = ['Year', 'Month', 'arrivals', 'sales', 'T_max', 'T_min', 'Rainfall', 'Area', 'Production', 'Teja_price', 'LCA-334', 'G-274', 'BYD-335']
    if not all(col in data.columns for col in required_columns):
        return "Required columns are missing from the dataset."

    # Create 'ds' column
    data['ds'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1))

    # Define varieties
    varieties = ['Teja_price', 'LCA-334', 'G-274', 'BYD-335']
    figures = []
    all_forecasts = []
    accuracy_summary = []

    for variety in varieties:
        data['y'] = data[variety]
        
        # Drop rows with missing values
        data = data.dropna(subset=['y', 'ds', 'arrivals', 'sales', 'T_max', 'T_min', 'Rainfall', 'Area', 'Production'])
        
        if data.empty:
            print(f"No data available for {variety} after dropping NaNs.")
            continue

        # Fit the Prophet model
        model = Prophet(seasonality_mode='multiplicative', changepoint_prior_scale=10)
        model.add_regressor('arrivals')
        model.add_regressor('sales')
        model.add_regressor('T_max')
        model.add_regressor('T_min')
        model.add_regressor('Rainfall')
        model.add_regressor('Area')
        model.add_regressor('Production')
        model.fit(data[['ds', 'y', 'arrivals', 'sales', 'T_max', 'T_min', 'Rainfall', 'Area', 'Production']])

        # Create future DataFrame
        future = model.make_future_dataframe(periods=12, freq='M')
        future = future.merge(data[['ds', 'arrivals', 'sales', 'T_max', 'T_min', 'Rainfall', 'Area', 'Production']], on='ds', how='left')

        # Fill missing future regressors if necessary
        future.fillna(method='ffill', inplace=True)

        # Generate forecast
        forecast = model.predict(future)

        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='<b>Historical Price </b>', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='<b>Forecast Price </b>', line=dict(color='green')))



    # Update layout
        fig.update_layout(
       title=f'<b>{variety} Forecast</b>',
    xaxis=dict(
            title='<b>Month</b>',
            tickfont=dict(size=14, family='Arial Black', color='black')
        ),
        yaxis=dict(
            title='<b>Price (per Tons)</b>',
            tickprefix='<b>₹</b>',  # Adds bold Indian Rupee symbol as prefix
            tickfont=dict(size=14, family='Arial Black', color='black')
        ),
            hovermode='x unified',
            height=600
        )
        # Convert figure to JSON
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        figures.append(graphJSON)

        # Prepare accuracy metrics
        actual_data = data[['ds', 'y']].copy()
        actual_data.rename(columns={'y': 'actual'}, inplace=True)
        merged = forecast.merge(actual_data, on='ds', how='left')
        merged['error'] = merged['actual'] - merged['yhat']
        merged['abs_error'] = merged['error'].abs()
        merged['abs_percentage_error'] = (merged['abs_error'] / merged['actual']) * 100

        merged['accuracy'] = 100 - merged['abs_percentage_error']
        merged_sorted = merged.sort_values(by='ds', ascending=False)
        forecast_table = merged_sorted[['ds', 'actual', 'yhat', 'error', 'abs_error', 'abs_percentage_error','accuracy']]
        forecast_table['variety'] = variety
        all_forecasts.append(forecast_table)

        avg_accuracy = merged['accuracy'].mean()
        accuracy_summary.append({'variety': variety, 'avg_accuracy': avg_accuracy})

    # Concatenate forecasts
    all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    forecast_data = all_forecasts_df.to_dict(orient='records')

    # Render the template
    return render_template('forecasting12.html', figures=figures, forecast_data=forecast_data, accuracy_summary=accuracy_summary)


@app.route('/day')
def day():
    # Load your data
    data = pd.read_csv('S.csv')
    print("Columns in the dataset:", data.columns)

    results = {}

    if 'Date' in data.columns and 'Modal' in data.columns:
        # Process date and handle invalid dates
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y', errors='coerce')
        invalid_dates = data[data['Date'].isna()]
        print("Invalid dates detected:\n", invalid_dates)
        data = data.dropna(subset=['Date'])
        data = data.rename(columns={'Date': 'ds', 'Modal': 'y'})

        # Replace 'NO TRANSACTIONS' in 'MIN' with 'MAX' value
        data['MIN'] = data.apply(
            lambda row: row['MAX'] if row['MIN'] == 'NO TRANSACTIONS' else row['MIN'], axis=1
        )

        # Remove 'NO TRANSACTIONS' rows in 'y' and process holidays
        data['y'] = data['y'].astype(str)
        holiday_rows = data[data['y'].str.contains("NO TRANSACTIONS", na=False)].copy()
        holiday_rows['holiday'] = holiday_rows['y'].str.replace("NO TRANSACTIONS DUE TO ", "").str.strip()
        holiday_dates = holiday_rows[['holiday', 'ds']]
        data = data[~data['y'].str.contains("NO TRANSACTIONS", na=False)].copy()

        data['y'] = pd.to_numeric(data['y'], errors='coerce')
        data = data.dropna(subset=['y'])

        holidays = holiday_dates.dropna().drop_duplicates()
        holidays['lower_window'] = 0
        holidays['upper_window'] = 0

        # Initialize Prophet model
        model = Prophet(
            holidays=holidays,
            changepoint_prior_scale=10,  # Lower value to reduce overfitting
            seasonality_prior_scale=8,    # Adjust for smoother seasonality
            holidays_prior_scale=10,
            n_changepoints=50,
            daily_seasonality=True      # Adjust this as needed
        )
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_seasonality(name='weekly', period=7, fourier_order=3)
        model.fit(data)

        # Predict future data
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)
        forecast['yhat_smoothed'] = forecast['yhat'].rolling(window=7).mean()

        # Calculate errors and accuracy metrics
        forecast_data = forecast[forecast['ds'].isin(data['ds'])]
        mae = mean_absolute_error(data['y'], forecast_data['yhat'])
        mean_actual = np.mean(data['y'])
        accuracy = 1 - (mae / mean_actual)
        mape = np.mean(np.abs((data['y'] - forecast_data['yhat']) / data['y'])) * 100

        # Prepare results dictionary for output
        results['mae'] = mae
        results['accuracy'] = accuracy * 100

        # Create the detailed table data
        detailed_data = data[['ds', 'y']].merge(
            forecast_data[['ds', 'yhat']], on='ds', how='left'
        )
        detailed_data['Error (Per Tons)'] = detailed_data['y'] - detailed_data['yhat']
        detailed_data['Abs Error (Per Tons)'] = detailed_data['Error (Per Tons)'].abs()
        detailed_data['Abs Percentage Error (Per Tons)'] = (detailed_data['Abs Error (Per Tons)'] / detailed_data['y']) * 100
        detailed_data['Accuracy Percentage (Per Tons)'] = 100 - detailed_data['Abs Percentage Error (Per Tons)']

        # Convert the detailed data to a dictionary for rendering in the table
        results['detailed_table'] = detailed_data.to_dict('records')
        future_forecast = forecast[['ds', 'yhat']].tail(365)  # Get the forecasted values for the next 365 days
        future_forecast = future_forecast.rename(columns={'ds': 'Date', 'yhat': 'Forecasted Price (Per Qtl)'})
        results['future_forecast'] = future_forecast.to_dict('records')
        # Cross-validation metrics
        cv_results = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')
        performance = performance_metrics(cv_results)
        print("Cross-validation metrics:\n", performance[['horizon', 'mae', 'rmse']].head())

        # Plot the data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='<b>Historical Price</b>', marker=dict(color='orange', size=6)))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_smoothed'], mode='lines', name='<b>Forecast Price</b>', line=dict(color='black')))

                fig.add_annotation(
                    x=0.5, y=0.95, xref="paper", yref="paper",
                    text=f"MAE: {mae:.2f} <br> Accuracy: {accuracy * 100:.2f}%",
                    showarrow=False, font=dict(size=14, color="black"),
                    align="center", bgcolor="rgba(255, 255, 255, 0.8)", borderpad=4
                )

                fig.update_layout(
                    title='<b>Teja Variety (2011-01 to 2018-12</b>',
                 xaxis=dict(
                    title='<b>Date</b>',
                    tickfont=dict(size=14, family='Arial Black', color='black')
                ),
                yaxis=dict(
                    title='<b>Teja Price (Per Qtl)</b>',
                    tickprefix='<b>₹</b>',  # Adds bold Indian Rupee symbol as prefix
                    tickfont=dict(size=14, family='Arial Black', color='black')
                ),
                    hovermode='x unified',
                    height=600
                )
                fig.write_html("static/forecast_plot.html")  # Save plot to a static folder
            except Exception as e:
                print("An error occurred in graph display:", e)

        # Render the results to the HTML template with the detailed table data
        return render_template('d_results.html', results=results)

    else:
        return render_template('error.html', message="Columns 'Date' and 'Modal' are missing in the dataset.")

@app.route('/forecastingls')
def forecastingls():
    data = pd.read_excel('bb.xlsx')

    # Check for 'Year' and 'Month' columns and create 'ds' column
    if 'Year' in data.columns and 'Month' in data.columns:
        data['ds'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1))
    else:
        raise ValueError("Year and Month columns are required.")

    varieties = ['Teja_price', 'LCA-334', 'G-274', 'BYD-335']
    figures = []  # List to hold individual figures for each variety

    for variety in varieties:
        data['y'] = data[variety]  # Set the target variable for the current variety

        # Drop rows with missing or NaN values
        data = data.dropna(subset=['y', 'ds'])

        # Ensure there is data to work with
        if data.empty:
            print(f"No data available for {variety} after dropping NaNs.")
            continue  # Skip to the next variety if there's no data

        # Normalize the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[['y']])

        # Create sequences
        def create_sequences(data, seq_length):
            xs, ys = [], []
            for i in range(len(data)-seq_length):
                x = data[i:i+seq_length]
                y = data[i+seq_length]
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)

        seq_length = 12  # Example sequence length (12 months)
        X, y = create_sequences(scaled_data, seq_length)

        # Split into train and test sets
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
        model.add(LSTM(50))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

        # Make predictions
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        # Prepare forecast DataFrame
        forecast_dates = pd.date_range(start=data['ds'].iloc[-1], periods=12, freq='M')
        forecast_df = pd.DataFrame({'ds': forecast_dates, 'yhat': predictions.flatten()})

        # Create a new figure for the current variety
        fig = go.Figure()

        # Add historical price trace
        trace1 = go.Scatter(
            x=data['ds'],
            y=data['y'],
            mode='lines',
            name='Historical Price (per Tons)',
            line=dict(color='blue')
        )
        fig.add_trace(trace1)

        # Add forecast trace
        trace2 = go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat'],
            mode='lines',
            name='Forecast Price (per Tons)',
            line=dict(color='green')
        )
        fig.add_trace(trace2)

        # Update layout for the individual figure
        fig.update_layout(
            title=f'<b>Forecasting for {variety}</b>',
            xaxis_title='Date',
            yaxis_title='Price (per Tons)',
            hovermode='x unified',
            height=500  # Adjust height for individual figures as needed
        )

        # Convert the figure to JSON for rendering in the template
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Append the JSON representation to the figures list
        figures.append(graphJSON)

    # Render the template with the list of figure JSONs
    return render_template('forecastingls.html', figures=figures)


@app.route('/prediction')
def prediction():
    return render_template('prediction.html')  # Serve the Prediction page


df = pd.read_excel('aa.xlsx')
df.columns = df.columns.str.strip().str.replace('\n', '').str.replace('\r', '')

# Preprocess the data
df['Year-Week'] = pd.to_datetime(df['Year-Week'] + '-1', format='%Y-W%W-%w')
df['Year'] = df['Year-Week'].dt.year
df['Week'] = df['Year-Week'].dt.isocalendar().week





if __name__ == '__main__':
 app.run(debug=True, port=4996)
