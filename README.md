# Flight Profitability Prediction Streamlit App

## Overview
This Streamlit application is designed to predict flight profitability using machine learning. The app preprocesses uploaded datasets, trains a Random Forest Regressor model, and provides insights through various performance metrics and KPIs.

## Features
1. **Model Performance Tab:**
   - Trains a Random Forest Regressor on the uploaded dataset.
   - Displays R² Score and Mean Absolute Error (MAE) to evaluate the model's performance.
   
2. **KPI Visualization Tab:**
   - Provides key performance indicators based on the predicted data.
   - Includes insights on cost efficiency, fleet utilization, load factor, and profit margin ratios.

3. **Model Results Tab:**
   - Displays predictions and insights based on the model's output.
   - Offers comparisons between predicted and actual profitability.

4. **Dataset Testing Tab:**
   - Allows users to test another dataset with the same structure.
   - Uses the pre-trained model to predict profitability for new data.

## Installation and Usage
### Prerequisites:
- Python 3.x
- Streamlit
- Pandas
- Scikit-learn
- Joblib
- NumPy

### Steps to Run the Application:
1. Install dependencies:
   ```bash
   pip install streamlit pandas scikit-learn joblib numpy
   ```
2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Upload a dataset (CSV or Excel) through the sidebar.
4. Navigate between tabs to explore model performance, KPIs, results, and dataset testing.

## Data Preprocessing
- Handles missing values by filling numeric columns with their median values.
- Feature engineering includes:
  - Cost efficiency calculation
  - Utilization efficiency computation
  - Fleet and load efficiency metrics
  - Time-based transformations (season, delay category)
  
## Model Training
- Uses **RandomForestRegressor** with optimized hyperparameters:
  - `n_estimators=200`
  - `max_depth=10`
  - `min_samples_split=5`
  - `min_samples_leaf=2`
- StandardScaler is applied to normalize features.
- The trained model and scaler are saved using Joblib for future use.

## Expected Dataset Format
Ensure that the dataset includes the following columns:
- `Operating Cost (USD)`, `Revenue (USD)`, `Profit (USD)`
- `Aircraft Utilization (Hours/Day)`, `Turnaround Time (Minutes)`
- `Net Profit Margin (%)`, `Fleet Availability (%)`, `Load Factor (%)`
- `Scheduled Departure Time`, `Actual Departure Time`
- `Delay (Minutes)`

If certain columns are missing, default values are used to maintain feature consistency.

## Future Enhancements
- Implement feature selection for better interpretability.
- Improve visualization of KPIs and model results.
- Add user-defined model tuning options.

## Contributors
Developed as part of a data analytics project to predict airline profitability and optimize fleet efficiency.


