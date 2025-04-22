def train_exponential_regression_model(useful_data):
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
    from scipy.optimize import curve_fit
    import numpy as np
    import pandas as pd
    import plotly.express as px

    # Define the exponential function
    def exponential_func(x, a, b, c):
        return a * np.exp(b * (x - 1900)) + c

    X = useful_data['Year'].values
    y = useful_data['Population'].values

    # Fit the exponential model
    params, _ = curve_fit(exponential_func, X, y, p0=[1, 0.01, 0], maxfev=10000)
    a, b, c = params

    # Split data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Make predictions
    y_pred = exponential_func(X, a, b, c)
    y_test_pred = exponential_func(X_test, a, b, c)

    # Calculate metrics
    rmse = root_mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # Predict population for the next 5 years
    future_years = np.array([X[-1] + i for i in range(1, 6)])
    future_predictions = exponential_func(future_years, a, b, c)

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'Year': np.concatenate([X_train, X_test, future_years]),
        'Population': np.concatenate([y_train, y_test_pred, future_predictions]),
        'Type': ['Historical'] * len(X_train) + ['Predicted'] * len(X_test) + ['Future'] * len(future_years)
    })

    # Add actual test values to the DataFrame
    actual_test_data = pd.DataFrame({
        'Year': X_test,
        'Population': y_test,
        'Type': ['Historical'] * len(X_test)
    })

    plot_data = pd.concat([plot_data, actual_test_data], ignore_index=True)

    # Create the plot
    fig = px.scatter(plot_data, x='Year', y='Population', color='Type', 
                     title='Population Prediction using Exponential Regression', labels={'Population': 'Population', 'Year': 'Year'})

    return rmse, mae, r2, fig