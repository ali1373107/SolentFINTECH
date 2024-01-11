
import pandas as pd
import yfinance as yf
import datetime
import pmdarima as pm
from keras.src.layers import Dropout
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import seaborn as sns
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from prophet import Prophet
def getdata(startday,endday,selected_column):
    all_company_data = pd.DataFrame()

    company_tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "CMBS", "TSLA", "JPM", "WMT", "V", "PG",
                       "AAP", "GOOG", "IBM", "NFLX", "GS", "GM", "GE", "CSCO", "BA", "C",
                       "CVX", "PEP", "KO", "INTC", "MCD", "DIS", "HD", "IBM", "AXP", "MMM",
                       "CAT", "UNH", "MRK", "NKE", "VZ", "PFE", "RTX", "ULST", "TRV", "HON",
                       "HTUS", "LGOV", "RIGS", "HYEM", "FFIU", "VMBS", "IEF", "MBB", "SPMB", "SCHP",
                       "JMBS", "BIV", "MFLX", "JCPB", "FLMI", "FBND", "MMIN", "SCHR", "TDTF", "VGIT",
                       "SPTI", "UITB", "IEI", "IGEB", "TIPX", "IAGG", "GTO", "HTAB", "VCIT", "BAB",
                       "FCOR", "SKOR", "FCAL", "HMOP", "KORP", "HYHG", "GRNB", "TDTT", "AGZ", "FSMB",
                       "EMTL", "CMBS", "VTIP", "IGSB", "USTB", "SPTS", "PFFA", "SUSB", "VCSH", "STIP",
                       "LSST", "SCHO", "PBTP", "VGSH", "EMCB", "SPSB", "LMBS", "SLQD", "NEAR", "MUST"]
    #get data from yahoo finance
    for company in company_tickers:
        data = yf.download(company, start=startday, end=endday)
        if selected_column not in data.columns:
            print(f"Warning: {company} does not have the selected column {selected_column}")
            continue

        data = data[[selected_column]]
        data = data.rename(columns={selected_column: company})
        all_company_data = pd.concat([all_company_data, data[company]], axis=1)


    transformed_data = all_company_data.T
    if transformed_data.isna().any().any():
        print("There are missing values in the DataFrame.")

        # Fill missing values with the mean of each row
        transformed_data_filled = transformed_data.apply(lambda row: row.fillna(row.mean()), axis=1)

        # Check again for missing values after filling
        if transformed_data_filled.isna().any().any():
            print("There are still missing values in the DataFrame after filling.")
        else:
            print("No missing values in the DataFrame after filling.")

        transformed_data = transformed_data_filled

        # Check for the number of rows and columns
    print("Number of rows:", transformed_data.shape[0])
    print("Number of columns:", transformed_data.shape[1])
    transformed_data.to_csv('stock_data.csv')
    df = pd.read_csv('stock_data.csv')
    return transformed_data
def dimensionality_reduction(company_data):

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(company_data)
    # extract 10 principal components
    pca = PCA(n_components=10)
    pca_reduced_data = pca.fit_transform(scaled_data)
    print("Number of rows After PCA:", pca_reduced_data.shape[0])
    print("Number of Culomnss after PCA :", pca_reduced_data.shape[1])
    column_names = company_data.columns
    top_components_loadings = pca.components_[:10]
    top_columns_indices = [np.abs(loadings).argsort()[-1] for loadings in top_components_loadings]
    top_columns = column_names[top_columns_indices]
    print("Top 10 columns (days) after PCA:")
    print(top_columns)
    return pca_reduced_data


def clustring(data):
    os.environ["LOKY_MAX_CPU_COUNT"] = "4"
    kmeans = KMeans(n_clusters=4,init='random', n_init=10)

    clusters = kmeans.fit_predict(data)

    return clusters

def visualize_clusters(pca_reduced_data, clusters, mydata):
    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(pca_reduced_data[:, 0], pca_reduced_data[:, 1], c=clusters, cmap='viridis')

    for i, company in enumerate(mydata.index):
        plt.annotate(company, (pca_reduced_data[i, 0], pca_reduced_data[i, 1]))

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Scatter Plot of PCA 1 vs PCA 2 with Clusters')
    plt.legend(*scatter.legend_elements(), title='Clusters')
    plt.show()

def select_representative_stock(pca_reduced_data, clusters, mydata):
    # Find the centroi the clusters
    centroids = []
    for i in range(max(clusters) + 1):
        cluster_points = pca_reduced_data[clusters == i]
        centroid = cluster_points.mean(axis=0)
        centroids.append(centroid)

    # Find the stock closest to each centroid
    representative_stocks = []
    for centroid in centroids:
        distances = np.linalg.norm(pca_reduced_data - centroid, axis=1)
        closest_stock_index = np.argmin(distances)
        stock = mydata.index[closest_stock_index]  # Use index instead of columns

        if stock not in mydata.index:
            print(f"Error: {stock} is not present in the index.")
        else:
            entry = {"stock": stock, "centroid": centroid}
            representative_stocks.append(entry)

    return representative_stocks
def find_correlations(selected_stock, all_data, num_correlated=10):
    all_correlations = {}


        # Calculate correlations with the selected stock
    correlations = all_data.T.corrwith(all_data.T[selected_stock])

        # Sort correlations in descending order
    sorted_correlations = correlations.sort_values(ascending=False)

        # Exclude the first value (self-correlation) and get the top 10
    top_positive_correlations = sorted_correlations.iloc[1:].head(num_correlated)
    top_negative_correlations = sorted_correlations.tail(num_correlated)

        # Store correlations in a dictionary
    all_correlations[selected_stock] = {
        'top_positive': top_positive_correlations,
        'top_negative': top_negative_correlations
    }


    return all_correlations

def perform_eda(stock_data, stock_name):
    # Check if the index is in datetime format, if not, set it to a range of dates
    if not isinstance(stock_data.index, pd.DatetimeIndex):
        stock_data.index = pd.date_range(start='2023-1-1', periods=len(stock_data), freq='B')
        # Displaying the first few rows of the dataset
    print(stock_data.head())
    # Displaying general information about the dataset
    print(stock_data.info())

    # Summary statistics
    print(stock_data.describe())

    # Checking for missing values
    print(stock_data.isnull().sum())
    data = stock_data.dropna()# Example: Drop rows with missing values
    # Univariate Analysis
    plt.figure(figsize=(12, 6))
    data[stock_name].plot(label=stock_name, color='blue')
    plt.title(f'Closing Prices Over Time for {stock_name}')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()
    # Visualize the distribution of observations
    plt.figure(figsize=(10, 6))
    sns.histplot(data[stock_name], bins=30, kde=True, color='blue')
    plt.title(f'Distribution of {stock_name} Observations')
    plt.xlabel('Stock Price')
    plt.ylabel('Frequency')
    plt.show()
    # Investigate the change in distribution over intervals
    plt.figure(figsize=(16, 8))
    sns.boxplot(x=data.index.year, y=data[stock_name])
    plt.title(f'Change in Distribution of {stock_name} Over Years')
    plt.xlabel('Year')
    plt.ylabel('Stock Price')
    plt.show()
def visualize_correlations(correlations_result):
    for stock, correlations in correlations_result.items():
        # Exclude the first value (self-correlation) and get the top 10
        top_positive = correlations['top_positive'].iloc[1:]

        # Visualize top positive correlations
        plt.figure(figsize=(12, 6))
        top_positive.plot(kind='bar', color='g')
        plt.title(f"Top 10 Positively Correlated Stocks with {stock}")
        plt.xlabel("Stocks")
        plt.ylabel("Correlation")
        plt.show()

        # Visualize top negative correlations with inverted y-axis
        plt.figure(figsize=(12, 6))
        correlations['top_negative'].plot(kind='bar', color='r')
        plt.title(f"Top 10 Negatively Correlated Stocks with {stock}")
        plt.xlabel("Stocks")
        plt.ylabel("Correlation")
        plt.gca().invert_yaxis()  # Invert y-axis
        plt.show()

def train_lstm_model(data, n_steps=30, epochs=50, batch_size=1, test_size=0.2, visualize=True):
    # Transpose the data to have dates as rows and stocks as columns
    transposed_data = data.T

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(transposed_data)

    # Prepare data
    X, y = [], []
    for i in range(len(scaled_data) - n_steps):
        X.append(scaled_data[i:i + n_steps])
        y.append(scaled_data[i + n_steps])

    X, y = np.array(X), np.array(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, len(data))))
    model.add(Dropout(0.1))

    model.add(Dense(len(data)))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Inverse transform to get back the original scale
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)

    # Evaluate the model
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    print(f"Mean Squared Error on Test Set: {mse}")

    # Visualize results
    if visualize:
        plt.figure(figsize=(10, 6))
        plt.plot(y_test_inv, label='Actual Prices', color='blue')
        plt.plot(y_pred_inv, label='Predicted Prices', color='orange')
        plt.title('Stock Price Prediction with LSTM')
        plt.xlabel('Time Steps')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

    return mse

def train_arima_model(stock_data, stock_name, order=(5, 1, 0), test_size=0.2, visualize=True):
    # Ensure the index is in datetime format
    stock_data.index = pd.to_datetime(stock_data.index)

    # Prepare data
    data = stock_data[[stock_name]]

    # Train-test split
    train_size = int(len(data) * (1 - test_size))
    train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]

    # Fit ARIMA model using automatic hyperparameter selection
    model = pm.auto_arima(train_data, seasonal=True, m=12, suppress_warnings=True)
    fitted_model = model.fit(train_data)

    # Make predictions on the test set
    forecast_steps = len(test_data)
    y_pred, conf_int = fitted_model.predict(n_periods=forecast_steps, return_conf_int=True)

    # Evaluate the model
    mse = mean_squared_error(test_data, y_pred)
    print(f"Mean Squared Error on Test Set for {stock_name}: {mse}")

    # Visualize results
    if visualize:
        plt.figure(figsize=(10, 6))
        plt.plot(data, label='Actual Prices', color='blue')
        plt.plot(test_data.index, y_pred, label='Predicted Prices', color='orange')
        plt.fill_between(test_data.index, conf_int[:, 0], conf_int[:, 1], color='orange', alpha=0.2, label='95% Prediction Interval')
        plt.title(f'Stock Price Prediction with ARIMA - {stock_name}')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

    return mse

def train_prophet_models(stock_data,stock, training_period=252, forecast_period=30):
    prophet_models = {}


    data = stock_data[stock].reset_index()
    data = data.rename(columns={'index': 'ds', stock: 'y'})

    # Train the model on the specified training period
    train_data = data.tail(training_period)
    model = Prophet()
    model.fit(train_data)

    prophet_models[stock] = model

    # Plot the forecast for the specified forecast period
    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)
    fig = model.plot(forecast)
    plt.title(f"Prophet Forecast for {stock}")
    plt.show()

    return prophet_models

def train_svr_model(X, y, kernel='rbf', test_size=0.2, visualize=True):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    # Build the SVR model
    model = SVR(kernel=kernel)

    # Train the model
    model.fit(X_train, y_train.ravel())

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on Test Set: {mse}")

    # Visualize results
    if visualize:
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='Actual Prices', color='blue')
        plt.plot(y_pred, label='Predicted Prices', color='orange')
        plt.title('Stock Price Prediction with SVR')
        plt.xlabel('Time Steps')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

    return mse

# Function to handle EDA button click
def on_select(event):
    selected_stock = dropdown.get()
    selected_entry = next((entry for entry in rep_stocks_and_centroids if entry["stock"] == selected_stock), None)
    if selected_entry:
        # Call find_correlations with the selected stock
        correlations_result = find_correlations(selected_stock, mydata)

        # Enable the visualize button
        visualize_button['state'] = tk.NORMAL

        # Enable the EDA button
        eda_button['state'] = tk.NORMAL
        # Enable the LSTM button
        lstm_button['state'] = tk.NORMAL

        arima_button['state'] = tk.NORMAL

        prophet_button['state'] = tk.NORMAL

        SVR_button['state'] = tk.NORMAL

        V_clustering_button['state'] = tk.NORMAL
        # Save correlations_result to a global variable for access by the visualize function
        global correlations_data
        correlations_data = correlations_result



def visualize_button_click():
    # Call visualize_correlations with the saved correlations_data
    selected_stock = dropdown.get()
    correlations_result = find_correlations(selected_stock, mydata)
    visualize_correlations(correlations_result)


def eda_button_click():
    # Get the selected stock
    selected_stock = dropdown.get()
    selected_stock_data = mydata.loc[selected_stock].to_frame()

    # Perform EDA
    perform_eda(selected_stock_data, selected_stock)
def lstm_button_click():
    # Get the selected stock
    selected_stock = dropdown.get()

    # Extract the data for the selected stock
    selected_stock_data = mydata.loc[[selected_stock]]

    # Apply LSTM model
    mse = train_lstm_model(selected_stock_data, visualize=True)
def arima_button_click():
    # Get the selected stock
    stock_name = dropdown.get()

    # Extract the data for the selected stock
    stock_data = mydata.loc[[stock_name]].T  # Transpose to have dates as rows
    mse_result = train_arima_model(stock_data, stock_name, order=(5, 1, 0), test_size=0.2, visualize=True)


def prophet_button_click():
    selected_stock = dropdown.get()
    selected_entry = next((entry for entry in rep_stocks_and_centroids if entry["stock"] == selected_stock), None)

    if selected_entry:
        selected_stock_data = mydata.loc[[selected_stock]].T  # Transpose to have dates as rows
        train_prophet_models(selected_stock_data, selected_stock)

def svr_button_click():
    # Get the selected stock from the dropdown
    stock_name = dropdown.get()

    # Extract the closing price as the target variable
    y = mydata.loc[stock_name].values

    # Create a lag feature
    lag = 1
    X = np.roll(y, lag)
    X[:lag] = 0
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Call the train_svr_model function for each stock
    mse_result = train_svr_model(X_scaled, y, kernel='rbf', test_size=0.2, visualize=True)

def v_clustring_button_click():
    # Call visualize_correlations with the saved correlations_data
    mydata = getdata(startday, endday, selected_column)
    pca_reduced_data = dimensionality_reduction(mydata)
    clusters = clustring(pca_reduced_data)
    visualize_clusters(pca_reduced_data, clusters, mydata)
# Define the start and end dates and selected column
startday = datetime.datetime(2023, 1, 1)
endday = datetime.date.today()
selected_column = "Close"

# Get financial data and perform dimensionality reduction and clustering
mydata = getdata(startday, endday, selected_column)
pca_reduced_data = dimensionality_reduction(mydata)
clusters = clustring(pca_reduced_data)
# Get representative stocks and centroids
rep_stocks_and_centroids = select_representative_stock(pca_reduced_data, clusters, mydata)

# Create the main window
root = tk.Tk()
root.title("Stock Selector")

# Create a label
label = ttk.Label(root, text="Select a stock:")
label.grid(row=0, column=0, padx=10, pady=10)

# Create a dropdown menu with representative stocks
representative_stocks = [entry["stock"] for entry in rep_stocks_and_centroids]  # Extract stock names
dropdown = ttk.Combobox(root, values=representative_stocks, state="readonly")
dropdown.grid(row=0, column=1, padx=10, pady=10)
dropdown.set("Select a stock")  # Default text

# Create a button for visualization
visualize_button = ttk.Button(root, text="Visualize Correlations", command=visualize_button_click, state=tk.DISABLED)
visualize_button.grid(row=1, column=0, columnspan=2, pady=10)


# Create a button for EDA
eda_button = ttk.Button(root, text="Perform EDA", command=eda_button_click, state=tk.DISABLED)
eda_button.grid(row=2, column=0, columnspan=2, pady=10)

lstm_button = ttk.Button(root, text="Apply LSTM", command=lstm_button_click, state=tk.DISABLED)
lstm_button.grid(row=3, column=0, columnspan=2, pady=10)

arima_button = ttk.Button(root, text="Apply ARIMA", command=arima_button_click, state=tk.DISABLED)
arima_button.grid(row=4, column=0, columnspan=2, pady=10)

prophet_button = ttk.Button(root, text="Apply Prophet Model", command=prophet_button_click, state=tk.DISABLED)
prophet_button.grid(row=5, column=0, columnspan=2, pady=10)

SVR_button = ttk.Button(root, text="Apply SVR", command=svr_button_click, state=tk.DISABLED)
SVR_button.grid(row=6, column=0, columnspan=2, pady=10)

V_clustering_button = ttk.Button(root, text="Visualize clustring", command=v_clustring_button_click, state=tk.NORMAL)
V_clustering_button.grid(row=7, column=0, columnspan=2, pady=10)
# Bind the event handler to the dropdown selection
dropdown.bind("<<ComboboxSelected>>", on_select)

# Run the GUI
root.mainloop()