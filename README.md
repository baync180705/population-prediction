# Population Prediction Project

This project is designed to predict the population of India using various machine learning models, including ARIMA, Linear Regression, XGBoost, and Exponential Regression. The project uses historical population data to train these models and provides a Streamlit-based web interface for users to interact with the models and view predictions.

## Features
- Predict population using ARIMA, Linear Regression, XGBoost, and Exponential Regression models.
- Visualize model performance metrics such as RMSE, MAE, and R2 Score.
- Interactive web interface built with Streamlit.

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd population-prediction
   ```

2. **Install Dependencies**
   Make sure you have Python installed on your system. Then, install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Data**
   Ensure the dataset `indian population new.csv` is located in the `data/` directory. This dataset should contain the following columns:
   - `Year`
   - `Population`
   - `% Increase in Population`

## Running the Streamlit App

1. Navigate to the `models/` directory:
   ```bash
   cd models
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open the provided URL in your web browser to interact with the app.

## Project Structure
- `data/`: Contains the dataset used for training and predictions.
- `models/`: Contains the implementation of various machine learning models and the Streamlit app.
- `requirements.txt`: Lists the Python dependencies required for the project.

## Notes
- Ensure that all required Python packages are installed before running the app.
- The app is designed to work on Windows systems; paths may need adjustment for other operating systems.