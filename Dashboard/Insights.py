def show_insights():
    
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import zipfile

    zip_path = "/mount/src/airline_industry_analysis/Dashboard/season_map.zip"  # Path to your ZIP file
    csv_filename = "season_map.csv"  # Name of the CSV inside the ZIP

    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(csv_filename) as f:
            season_data = pd.read_csv(f)
       
    # Load Data
    df = pd.read_csv("/mount/src/airline_industry_analysis/Dashboard/predicted_values.csv")
    model_results = pd.read_csv("/mount/src/airline_industry_analysis/Dashboard/model_results.csv")
    # season_data = pd.read_csv("/mount/src/airline_industry_analysis/Dashboard/season_map.csv")
    
    # Standardizing Column Names
    df.columns = df.columns.str.strip()
    season_data.columns = season_data.columns.str.strip()
    
    # Mapping numeric delay categories to descriptive labels
    delay_mapping = {0: "Short", 1: "Medium", 2: "Long"}
    season_data["Delay_Category"] = season_data["Delay_Category"].map(delay_mapping)
    
    # # Streamlit App Configuration
    # st.set_page_config(page_title="Airline Industry KPI Dashboard", layout="wide")
    
    
    # Custom CSS for Styling with Background Image
    st.markdown(
        """
        <style>
            .stApp {
                background: url('https://www.omnevo.net/fileadmin/Omnevo/images/main_navigation/insights/2022/adst_270865104_airline-airplane-sky-sunset-long.jpg') no-repeat center center fixed;
                background-size: cover;
            }
            .main { background-color: rgba(255, 255, 255, 0.85); padding: 20px; border-radius: 10px; }
            h1, h2, h3 { color: #2C3E50; }
            div[data-testid="stMetric"] {
                border-radius: 8px;
                background-color: #ffffff;
                padding: 10px;
                margin: 5px;
                box-shadow: 3px 3px 10px rgba(0,0,0,0.1);
            }
            .st-emotion-cache-16idsys p { 
                font-size: 18px;
                font-weight: bold;
                color: #2C3E50;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    
    # **KPI Dashboard**
    st.title("📊 Airline Industry KPI Dashboard")
    
    # **1️⃣ Key Performance Metrics**
    st.subheader("✈️ Key Performance Metrics")
    
    # Financial Metrics
    total_flights = df.shape[0]
    avg_profit = df["Predicted Profit (USD)"].mean()
    total_revenue = season_data["Revenue (USD)"].sum() if "Revenue (USD)" in season_data.columns else "N/A"
    total_cost = season_data["Operating Cost (USD)"].sum() if "Operating Cost (USD)" in season_data.columns else "N/A"
    
    # Operational Metrics
    load_factor = season_data["Load Factor (%)"].mean() if "Load Factor (%)" in season_data.columns else "N/A"
    fleet_utilization = season_data["Fleet Availability (%)"].mean() if "Fleet Availability (%)" in season_data.columns else "N/A"
    
    # Displaying Metrics with Icons
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✈️ Total Flights Analyzed", total_flights)
    col2.metric("💰 Average Predicted Profit", f"${avg_profit:,.2f}")
    col3.metric("📈 Average Load Factor (%)", f"{load_factor:.2f}%" if load_factor != "N/A" else "N/A")
    col4.metric("🚀 Average Fleet Utilization (%)", f"{fleet_utilization:.2f}%" if fleet_utilization != "N/A" else "N/A")
    
    col5, col6 = st.columns(2)
    col5.metric("💵 Total Revenue", f"${total_revenue:,.2f}" if total_revenue != "N/A" else "N/A")
    col6.metric("📊 Total Operating Cost", f"${total_cost:,.2f}" if total_cost != "N/A" else "N/A")
    
    
    # **KPI Visualizations**
    st.subheader("📊 KPI Insights")
    
    # 1️⃣ Distribution of Predicted Profit
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(df["Predicted Profit (USD)"], bins=30, kde=True, ax=ax, color="blue")
    ax.set_title("Distribution of Predicted Profit")
    ax.set_xlabel("Predicted Profit (USD)")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    st.pyplot(fig)
    
    # Monthly Profit Trends
    if "Actual_Month" in season_data.columns and "Profit (USD)" in season_data.columns:
        monthly_profit = season_data.groupby("Actual_Month")["Profit (USD)"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.lineplot(x=monthly_profit["Actual_Month"], y=monthly_profit["Profit (USD)"], marker="o", ax=ax)
        ax.set_title("Monthly Profit Trends")
        fig.patch.set_alpha(0)  # Make figure background transparent
        ax.patch.set_alpha(0)  # Make plot area background transparent
        st.pyplot(fig)
    
    # **Seasonal Insights**
    st.subheader("🌦️ Seasonal Performance Insights")
    
    # Seasonal Profitability Trends
    if "Actual_Season" in season_data.columns and "Profit (USD)" in season_data.columns:
        season_profit = season_data.groupby("Actual_Season")["Profit (USD)"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(x=season_profit["Actual_Season"], y=season_profit["Profit (USD)"], palette="coolwarm", ax=ax)
        ax.set_title("Average Profit by Season")
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        st.pyplot(fig)
    
    # Seasonal Flight Delays
    if "Actual_Season" in season_data.columns and "Delay_Category" in season_data.columns:
        season_delays = season_data.groupby("Actual_Season")["Delay_Category"].value_counts().unstack().fillna(0)
        fig, ax = plt.subplots(figsize=(6, 3))
        season_delays.plot(kind="bar", stacked=True, colormap="viridis", ax=ax)
        ax.set_title("Flight Delays by Season")
        ax.set_ylabel("Number of Delays")
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        st.pyplot(fig)
    
    # Revenue & Operating Cost by Season
    if "Actual_Season" in season_data.columns and "Revenue (USD)" in season_data.columns:
        season_revenue_cost = season_data.groupby("Actual_Season")[["Revenue (USD)", "Operating Cost (USD)"]].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 3))
        season_revenue_cost.set_index("Actual_Season").plot(kind="bar", colormap="Set2", ax=ax)
        ax.set_title("Revenue vs. Operating Cost by Season")
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        st.pyplot(fig)
    
    # Fleet Utilization by Season
    if "Actual_Season" in season_data.columns and "Fleet Availability (%)" in season_data.columns:
        season_fleet_utilization = season_data.groupby("Actual_Season")["Fleet Availability (%)"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(x=season_fleet_utilization["Actual_Season"], y=season_fleet_utilization["Fleet Availability (%)"], marker="o", ax=ax)
        ax.set_title("Fleet Utilization by Season")
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        st.pyplot(fig)
    
    # **Key Business Insights**
    st.subheader("🔍 Key Business Insights")
    
    insights = [
        "✈️ **Fuel Cost & Route Distance Correlation**: Longer routes generally lead to higher fuel costs, directly impacting airline profitability.",
        "📉 **Load Factor Impact**: Routes with **<70% seat occupancy** struggle with profitability, necessitating pricing optimizations.",
        "⏳ **Delay Impact on Costs**: Delays of **>30 minutes** increase flight costs by **15-20%** due to crew overtime and operational inefficiencies.",
        "📊 **Revenue per Available Seat Kilometer (RASK) Optimization**: Efficient fleet utilization plays a critical role in maintaining profitability.",
        "🌦️ **Seasonal Revenue Variability**: Revenue fluctuates by **up to 40%** between peak and off-peak seasons, making dynamic pricing crucial.",
        "✈️ **Fleet Utilization vs. Profitability**: Airlines with **>80% fleet utilization** tend to achieve **higher profit margins**, while underutilization              increases fixed costs per flight.",
        "🏆 **Operational Benchmarking**: Airlines with **efficient turnarounds (<45 min for short-haul, <90 min for long-haul)** improve fleet utilization and              reduce idle costs.",
    ]
    
    for insight in insights:
        st.write(insight)
    
    st.markdown("""<div style='text-align: center; padding: 10px;'>
                    <small>© 2025 Flight Analytics Dashboard</small>
                    </div>""", unsafe_allow_html=True)
    st.markdown("📌 *This dashboard provides data-driven insights for airline revenue optimization and operational efficiency.*")











