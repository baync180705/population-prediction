def train_arima_model(useful_data):
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
    import pandas as pd
    import plotly.express as px

    X_train = useful_data['Population'][:70]
    X_test = useful_data['Population'][70:]

    train_const = [x for x in X_train]
    pred_list = []

    for j in range(len(X_test)):
        model = ARIMA(train_const, order=(3, 0, 3))
        model_fit = model.fit()
        output = model_fit.forecast()
        pred = output[0]
        pred_list.append(pred)
        train_const.append(X_test.iloc[j])

    rmse = root_mean_squared_error(X_test, pred_list)
    mae = mean_absolute_error(X_test, pred_list)
    r2 = r2_score(X_test, pred_list)

    # Adjust indexes by adding 1950
    X_test.index = X_test.index + 1950
    X_train.index = X_train.index + 1950

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'Year': list(X_train.index) + list(X_test.index),
        'Population': list(X_train) + pred_list,
        'Type': ['Historical'] * len(X_train) + ['Predicted'] * len(pred_list)
    })

    # Add actual test values to the DataFrame
    actual_test_data = pd.DataFrame({
        'Year': X_test.index,
        'Population': X_test.values,
        'Type': ['Historical'] * len(X_test)
    })

    plot_data = pd.concat([plot_data, actual_test_data], ignore_index=True)

    # Predict population for the next 5 years
    future_years = [X_test.index[-1] + i for i in range(1, 6)]
    future_predictions = []

    for _ in range(5):
        model = ARIMA(train_const, order=(3, 0, 3))
        model_fit = model.fit()
        output = model_fit.forecast()
        pred = output[0]
        future_predictions.append(pred)
        train_const.append(pred)

    # Add future predictions to the DataFrame
    future_data = pd.DataFrame({
        'Year': future_years,
        'Population': future_predictions,
        'Type': ['Future'] * len(future_predictions)
    })

    plot_data = pd.concat([plot_data, future_data], ignore_index=True)

    # Create the plot
    fig = px.scatter(plot_data, x='Year', y='Population', color='Type', 
                     title='Population Prediction (Including Future)', labels={'Population': 'Population', 'Year': 'Year'})

    return rmse, mae, r2, fig
