def show_static_model_info():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import joblib
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    df = pd.read_csv("predicted_values.csv")
    model_results = pd.read_csv("model_results.csv")
    # Load dataset
    @st.cache_data
    def load_data():
        df = pd.read_excel("Aviation_KPIs_Dataset.xlsx")
        return df
    
    df = load_data()
    
    # Feature Engineering
    def feature_engineering(df):
        df['utilization_efficiency'] = df['Aircraft Utilization (Hours/Day)'] / (df['Turnaround Time (Minutes)'] + 1)
        df['profit_margin_ratio'] = df['Net Profit Margin (%)'] / 100
        df['cost_efficiency'] = df['Operating Cost (USD)'] / (df['Revenue (USD)'] + 1)
        df['fleet_utilization'] = df['Fleet Availability (%)'] * df['Aircraft Utilization (Hours/Day)']
        df['load_efficiency'] = df['Load Factor (%)'] * df['Fleet Availability (%)']
        return df
    
    df = feature_engineering(df)
    
    # Date processing
    df['Scheduled_Month'] = df['Scheduled Departure Time'].dt.month
    df['Actual_Month'] = df['Actual Departure Time'].dt.month
    df.drop(columns=['Scheduled Departure Time', 'Actual Departure Time'], inplace=True)
    
    season_mapping = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall"
    }
    
    df["Actual_Season"] = df["Actual_Month"].map(season_mapping)
    df["Scheduled_Season"] = df["Scheduled_Month"].map(season_mapping)
    
    le = LabelEncoder()
    df["Actual_Season"] = le.fit_transform(df["Actual_Season"])
    df["Scheduled_Season"] = le.fit_transform(df["Scheduled_Season"])
    
    df.drop(columns=['Scheduled_Month', 'Actual_Month', 'Delay (Minutes)'], inplace=True)
    
    # Train-Test Split
    X = df.drop(columns=["Profit (USD)", "Flight Number", "Revenue (USD)", "Operating Cost (USD)", "Revenue per ASK", "Cost per ASK"], errors="ignore")
    y = df["Profit (USD)"]
    X.fillna(X.median(numeric_only=True), inplace=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Save or load trained model
    @st.cache_resource
    def load_trained_model():
        try:
            model = joblib.load("trained_model.pkl")
        except FileNotFoundError:
            model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
            model.fit(X_train, y_train)
            joblib.dump(model, "trained_model.pkl")
        return model
    
    model = load_trained_model()
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Streamlit App Layout
    st.markdown("<h1 style='text-align: center;'>📊 Aviation Profit Prediction Model Implementation </h1>", unsafe_allow_html=True)
    # col1, col2,col3 = st.columns(1)
    # ----------------------- Model Lifecycle Flowchart -----------------------
    import streamlit as st
    
    # with col2:
    st.markdown("<h2 style='text-align: center;'>📌 Model Flowchart</h2>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = ["Data Preprocessing", "Feature Engineering", "Scaling", "Model Training", "Evaluation"]
    y_pos = np.arange(len(steps))
    
    ax.barh(y_pos, [1, 2, 3, 4, 5], color="skyblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(steps)
    ax.set_xlabel("Steps in Model Flow")
    ax.set_title("Model Lifecycle")
    st.pyplot(fig)
    col1, col2,col3 = st.columns(3)
    with col1:
         st.markdown("<h2 style='text-align: center;'>📌 Feature Engineering </h2>", unsafe_allow_html=True)
    
         st.markdown("""
         - *Utilization Efficiency*: Ratio of aircraft utilization to turnaround time.
         - *Profit Margin Ratio*: Measures overall profitability.
         - *Cost Efficiency*: Operating cost relative to revenue.
         - *Fleet Utilization*: A function of availability and utilization.
         - *Load Efficiency*: Represents flight occupancy rates.
         """)
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
         from PIL import Image
    
         st.header("📌 Feature Importance")
         image=Image.open("C:/Users/saksh/Airline_Analysis_Project/feature_importance.jpg")
         st.image(image)
        # feature_importance = model.feature_importances_
        # feature_names = X.columns
        # sorted_idx = np.argsort(feature_importance)[::-1]
        # plt.figure(figsize=(8, 6))
        # sns.barplot(x=feature_importance[sorted_idx],y=np.array(feature_names[sorted_idx]))
        # plt.xlabel("Feature Importance")
        # plt.ylabel("Features")
        # plt.title("Feature Importance of Model")
        # st.pyplot(plt)
         
    
     #Get feature importance
      
     
    with col2:
        st.subheader("Feature Correlation Heatmap")
    # Drop non-numeric columns before calculating correlation
        df_numeric = df.select_dtypes(include=[np.number])
    # Create columns for layout
    
    # Now plot the heatmap only for numerical columns
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_numeric.corr(), cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
        plt.title("Feature Correlation Matrix")
        st.pyplot(fig)
    col3, col4 = st.columns(2)
    with col3:
        st.header("🔹 Mapping & Encoding")
        st.markdown("""
        To convert *categorical data* into *numerical form*, we applied:
        - *Seasons Mapping:* Assigns numeric values to 'Winter', 'Summer', etc.
        - *Label Encoding:* Converts categorical flight delay categories into numbers.
        """)
        
    with col4:
        st.header("🔹 Why Scaling?")
        st.markdown("""
        Feature scaling ensures that numerical values are on the same scale, improving model performance.
        - *We applied Standard Scaling to normalize features.*
        """)
      
    
    
    col5, col6 = st.columns(2)
    with col5:
      st.header("🔹 Why Random Forest?")
      st.markdown("""
      We chose *Random Forest* because:
      - It handles both numerical & categorical features.
      - It reduces overfitting by averaging multiple trees.
      - It performs well on structured tabular data.
      """)
    with col6:
      st.header("🎯 *Model Performance Evaluation Metrics*")
      st.dataframe(model_results.style.format({"MAE": "{:.2f}", "R² Score": "{:.4f}"}))
      st.header(" 🎯 *Metric Interpretation:*")
      st.markdown("""
      - *MAE (Mean Absolute Error)*: Measures average prediction error. Lower is better.
      - *R² Score: Determines how well features explain profit variance. Closer to **1* is    ideal.""")