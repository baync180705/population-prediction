def train_linear_regression_model(useful_data):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
    import pandas as pd
    import numpy as np
    import plotly.express as px

    # Use only the last 10 values for training
    useful_data1 = useful_data.tail(10)

    X = useful_data1['Year'].values.reshape(-1, 1)
    y = useful_data1['Population'].values

    # Split the data into training and testing sets
    X_train, X_test = X[:7], X[7:]
    y_train, y_test = y[:7], y[7:]

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'Year': np.concatenate([X_train.flatten(), X_test.flatten()]),
        'Population': np.concatenate([y_train, y_pred]),
        'Type': ['Historical'] * len(y_train) + ['Predicted'] * len(y_pred)
    })

    # Include all years in the plot
    all_years = useful_data['Year'].values
    all_population = useful_data['Population'].values
    historical_data = pd.DataFrame({
        'Year': all_years,
        'Population': all_population,
        'Type': ['Historical'] * len(all_years)
    })

    plot_data = pd.concat([historical_data, plot_data], ignore_index=True)

    # Predict population for the next 5 years
    future_years = np.array([X_test[-1][0] + i for i in range(1, 6)]).reshape(-1, 1)
    future_predictions = model.predict(future_years)

    # Add future predictions to the DataFrame
    future_data = pd.DataFrame({
        'Year': future_years.flatten(),
        'Population': future_predictions.flatten(),
        'Type': ['Future'] * len(future_predictions)
    })

    plot_data = pd.concat([historical_data,plot_data, future_data], ignore_index=True)

    # Create the plot
    fig = px.scatter(plot_data, x='Year', y='Population', color='Type', 
                     title='Population Prediction (Including Future)', labels={'Population': 'Population', 'Year': 'Year'})

    return rmse, mae, r2, fig