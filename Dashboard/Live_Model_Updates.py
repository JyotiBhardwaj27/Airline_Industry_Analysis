def show_live_model_updates():
    import streamlit as st
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    
    # Feature Engineering Function
    def feature_engineering(df):
        df['utilization_efficiency'] = df['Aircraft Utilization (Hours/Day)'] / (df['Turnaround Time (Minutes)'] + 1)
        df['profit_margin_ratio'] = df['Net Profit Margin (%)'] / 100
        df['cost_efficiency'] = df['Operating Cost (USD)'] / (df['Revenue (USD)'] + 1)
        df['fleet_utilization'] = df['Fleet Availability (%)'] * df['Aircraft Utilization (Hours/Day)']
        df['load_efficiency'] = df['Load Factor (%)'] * df['Fleet Availability (%)']
        df['Scheduled_Month'] = df['Scheduled Departure Time'].dt.month
        df['Actual_Month'] = df['Actual Departure Time'].dt.month
        df.drop(columns=['Scheduled Departure Time', 'Actual Departure Time'], inplace=True)
        season_mapping = {
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Fall", 10: "Fall", 11: "Fall"
        }
        df['Actual_Season'] = df['Actual_Month'].map(season_mapping)
        df['Scheduled_Season'] = df['Scheduled_Month'].map(season_mapping)
        df['Delay_Category'] = np.where(df['Delay (Minutes)'] < 30, 'Short',
                                        np.where(df['Delay (Minutes)'] <= 60, 'Medium', 'Long'))
        df.drop(columns=['Scheduled_Month', 'Actual_Month', 'Delay (Minutes)'], inplace=True)
        df = pd.get_dummies(df, columns=['Actual_Season', 'Scheduled_Season', 'Delay_Category'], drop_first=True)
        return df
    
    # Function to train and save the model
    @st.cache_resource
    def train_and_save_model(df):
        df = feature_engineering(df)
        X = df.drop(columns=["Profit (USD)", "Flight Number", "Revenue (USD)", "Operating Cost (USD)", "Revenue per ASK", "Cost per ASK"], errors="ignore")
        y = df["Profit (USD)"]
        X.fillna(X.median(numeric_only=True), inplace=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
        model.fit(X_scaled, y)
        joblib.dump((model, scaler), 'flight_profit_model.pkl')
        return model, scaler
    
    # Function to load model
    @st.cache_resource
    def load_model():
        return joblib.load('flight_profit_model.pkl')
    
    # Page navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Model Performance"])
    
    # Upload file
    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    
        if page == "Model Performance":
            st.title("Flight Profitability Model Performance")
            model, scaler = train_and_save_model(df)
            df = feature_engineering(df)
            X = df.drop(columns=["Profit (USD)", "Flight Number", "Revenue (USD)", "Operating Cost (USD)", "Revenue per ASK", "Cost per ASK"], errors="ignore")
            y = df["Profit (USD)"]
            X.fillna(X.median(numeric_only=True), inplace=True)
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
    
            r2 = r2_score(y, predictions)
            mae = mean_absolute_error(y, predictions)
            st.metric("R² Score",f"{r2:.4f}")
            st.metric("Mean Absolute Error (MAE)",f"{mae:.2f}")