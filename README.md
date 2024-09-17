Stock Price Prediction using Deep Learning
Overview
This project implements a deep learning model to predict future stock prices using historical data. It compares the performance of different Recurrent Neural Network (RNN) architectures: Simple RNN, Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU).
Problem Statement
The goal is to predict the next 10 days of stock prices using a sequence of the past 60 days of price data. This is framed as a regression problem, with the closing price as the target variable.
Project Structure

Problem Definition: Clearly stating the objective and approach.
Data Collection: Using yfinance to download historical stock data (Google - GOOG).
Data Preprocessing:

Scaling the data using MinMaxScaler
Creating time series sequences


Model Building: Implementing three types of RNN models:

Simple RNN
LSTM
GRU


Model Training: Training each model on the prepared dataset.
Model Evaluation: Comparing model performance using Mean Squared Error (MSE) and Mean Absolute Error (MAE).
Visualization: Plotting actual vs predicted stock prices.

Requirements

Python 3.x
TensorFlow
NumPy
Pandas
Matplotlib
scikit-learn
yfinance

Installation
bashCopypip install tensorflow numpy pandas matplotlib scikit-learn yfinance
Usage



Run the Jupyter notebook or Python script:
bashCopyjupyter notebook Stock_price_prediction.ipynb
or
bashCopypython stock_price_prediction.py


Key Features

Utilizes yfinance for easy acquisition of historical stock data
Implements and compares three RNN architectures: Simple RNN, LSTM, and GRU
Uses sklearn for data preprocessing and evaluation metrics
Visualizes predictions against actual stock prices
Provides a comprehensive evaluation of model performance

Results
The project outputs:

Comparison table of MSE and MAE for each model (RNN, LSTM, GRU)
Visualizations of predicted vs actual stock prices for each model
Insights into which model performs best for this particular stock prediction task

Future Enhancements

Incorporate additional features like volume, open, high, and low prices
Implement hyperparameter tuning for model optimization
Explore ensemble methods for improved predictions
Extend the prediction horizon and test on different stocks or markets
Implement a simple web interface for real-time prediction
