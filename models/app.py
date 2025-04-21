import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from arima_streamlit import train_arima_model
from linearreg_streamlit import train_linear_regression_model

# Load data
data = pd.read_csv(r'C:\Users\Arnav Agarwal\Desktop\population-prediction\data\indian population new.csv')
useful_data = data[['Year', 'Population', '% Increase in Population']]

def fixdata(string):
    ans = "".join([i for i in string if i != ','])
    return int(ans)

useful_data['Population'] = useful_data['Population'].apply(fixdata)

# Streamlit app
st.title("Population Prediction using ARIMA")

# Add a dropdown for model selection
model = st.selectbox("Select a Model", ["ARIMA", "Linear Regression", "XGBoost", "Exponential Regression"])
st.write(f"You selected: {model}")

# Train ARIMA model when selected
if model == "ARIMA":
    rmse, mae, r2, fig = train_arima_model(useful_data)

    st.write(f"RMSE: {rmse}")
    st.write(f"MAE: {mae}")
    st.write(f"R2 Score: {r2}")
    st.plotly_chart(fig)

# Train Linear Regression model when selected
elif model == "Linear Regression":
    rmse, mae, r2, fig = train_linear_regression_model(useful_data)

    st.write(f"RMSE: {rmse}")
    st.write(f"MAE: {mae}")
    st.write(f"R2 Score: {r2}")
    st.plotly_chart(fig)

