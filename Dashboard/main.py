import sys
sys.path.append("/mount/src/airline_industry_analysis/Dashboard")

import streamlit as st
from Insights import show_insights
from Static_model_info import show_static_model_info
from Live_Model_Updates import show_live_model_updates

# # Set up the Streamlit app title
# st.set_page_config(page_title="Aviation Profit Prediction", layout="wide")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["📊 Insights", "📌 Static Model Info", "📡 Live Model Updates"])

# Tab 1: Insights
with tab1:
    show_insights()

# Tab 2: Static Model Information
with tab2:
    show_static_model_info()

# Tab 3: Live Model Updates
with tab3:
    show_live_model_updates()
