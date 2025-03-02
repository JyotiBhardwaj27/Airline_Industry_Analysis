# Aviation Profit Prediction Streamlit App

## Overview
The Aviation Profit Prediction Streamlit app is designed to provide insights and predictions on airline profitability. The app leverages a machine learning model (Random Forest Regressor) and offers rich visualizations and metrics through a user-friendly interface with three primary tabs: **Insights**, **Static Model Info**, and **Live Model Updates**.

## Application Structure
The application is organized into the following files:
- `main.py`: Launches the app and creates navigation tabs.
- `Insights.py`: Displays key performance indicators and insights.
- `Static_model_info.py`: Shows static information about the trained model.
- `Live_Model_Updates.py`: Provides live predictions and updates using the pre-trained model.


## Features
1. **Insights Tab:**
   - Provides visualizations and analysis of profitability metrics.
   - Displays KPIs such as cost efficiency, load factor, and fleet utilization.

2. **Static Model Info Tab:**
   - Shows model performance metrics (R² Score, Mean Absolute Error).
   - Describes the Random Forest Regressor model with key hyperparameters.

3. **Live Model Updates Tab:**
   - Allows testing of new datasets with the pre-trained model.
   - Displays predicted profitability and comparison with actual values.

## Data Preprocessing
- Fills missing numeric values with medians.
- Feature engineering steps include:
  - Cost and fleet utilization metrics.
  - Load factor calculations.
  - Seasonality and delay categorization.

## Model Details
- Model: **RandomForestRegressor**
- Key Hyperparameters:
  - `n_estimators=200`
  - `max_depth=10`
  - `min_samples_split=5`
  - `min_samples_leaf=2`
- Preprocessing: **StandardScaler** for feature normalization
- Model Persistence: Uses **Joblib** to save/load the model and scaler.

## Installation
### Prerequisites
Ensure you have Python 3.x and the required libraries installed.
```bash
pip install streamlit pandas scikit-learn joblib numpy
```

### Running the App
```bash
streamlit run main.py
```
- Upload a dataset (CSV or Excel) via the sidebar.
- Navigate through tabs to explore insights, model info, and predictions.

## Expected Dataset Format
Ensure the dataset includes these columns:
- `Operating Cost (USD)`, `Revenue (USD)`, `Profit (USD)`
- `Aircraft Utilization (Hours/Day)`, `Turnaround Time (Minutes)`
- `Net Profit Margin (%)`, `Fleet Availability (%)`, `Load Factor (%)`
- `Scheduled Departure Time`, `Actual Departure Time`, `Delay (Minutes)`

If certain columns are missing, default values are applied to maintain consistency.

## Future Enhancements
- Add feature selection options for model tuning.
- Improve visualization of live model updates.
- Enable user-defined model parameter adjustments.

## Contributors
Developed as part of a data analytics project focused on airline profitability and operational efficiency.

