def train_xgboost_model(useful_data):
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
    import pandas as pd
    import numpy as np
    import plotly.express as px

    # Prepare the data
    X = useful_data['Year'].values.reshape(-1, 1)
    y = useful_data['Population'].values

    # Split the data into training and testing sets
    train_size = 0.8
    split_idx = int(len(X) * train_size)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Predict population for the next 5 years
    future_years = np.array([X_test[-1][0] + i for i in range(1, 6)]).reshape(-1, 1)
    future_predictions = model.predict(future_years)

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'Year': np.concatenate([X.flatten(), future_years.flatten()]),
        'Population': np.concatenate([y, future_predictions]),
        'Type': ['Historical'] * len(X) + ['Future'] * len(future_years)
    })

    # Create the plot
    fig = px.scatter(plot_data, x='Year', y='Population', color='Type', 
                     title='Population Prediction using XGBoost', labels={'Population': 'Population', 'Year': 'Year'})

    return rmse, mae, r2, fig