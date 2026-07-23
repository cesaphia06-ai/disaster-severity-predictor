import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_resource
def load_pipeline():
    return joblib.load('disaster_severity_pipeline.pkl')

pipeline = load_pipeline()

st.title("🌋 Natural Disaster Impact & Severity Predictor")
st.write("Predict disaster severity levels using machine learning pipeline.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Input Disaster Attributes")
disaster_type = st.sidebar.selectbox("Disaster Type", ["Flood", "Earthquake", "Hurricane", "Tsunami", "Wildfire"])
location = st.sidebar.text_input("Location / Region", "Asia")
magnitude = st.sidebar.slider("Event Magnitude", 1.0, 10.0, 5.0)

# --- PREDICTION LOGIC ---
if st.sidebar.button("Predict Severity Level"):
    # Create input DataFrame with EXACT column names from training
    input_df = pd.DataFrame([{
        'Disaster_Type': disaster_type,
        'Location': location,
        'Magnitude': magnitude
    }])

    prediction = pipeline.predict(input_df)

    # 1. Main Output
    st.header("Model Prediction Output")
    st.success(f"Predicted Disaster Severity: **{prediction[0]}**")

    st.markdown("---")

    # Create 2 equal-width columns side-by-side
    col_summary, col_chart = st.columns([1, 1])

    # Left Column: Metrics
    with col_summary:
        st.subheader("📊 Input Profile Summary")
        st.metric("Disaster Type", disaster_type)
        st.metric("Region", location)
        st.metric("Magnitude Rating", f"{magnitude:.2f} / 10.0")

    # Right Column: Chart
    with col_chart:
        st.subheader("💡 Key Feature Drivers")
        st.caption("Event Magnitude accounts for ~76.6% of predictive weight.")
        
        drivers_df = pd.DataFrame({
            'Feature': ['Event Magnitude', 'Location / Region', 'Disaster Category'],
            'Importance (%)': [76.6, 13.4, 10.0]
        })

        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(data=drivers_df, x='Importance (%)', y='Feature', palette='Blues_r', ax=ax)
        ax.set_xlim(0, 100)
        
        st.pyplot(fig)
        
st.markdown("---")

with st.expander("ℹ️ About Model Predictions & Limitations"):
    st.markdown("""
    * **Primary Driver:** Physical event magnitude accounts for **~76.6%** of model predictive weight.
    * **Domain Insight:** Models trained strictly on physical attributes yield baseline accuracy (~33%). Actual financial and human loss depends heavily on unobserved local variables like population density, emergency infrastructure, and regional wealth.
    """)
    