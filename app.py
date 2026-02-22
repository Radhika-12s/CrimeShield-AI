import streamlit as st
import pandas as pd
import plotly.express as px
from src.data.data_loader import load_default_data, load_uploaded_data
from src.data.preprocessing import preprocess_data
from src.models.predict import predict_risk
from src.models.cvi_calculator import calculate_cvi
from src.utils.helpers import generate_report

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CrimeShield AI",
    page_icon="🛡",
    layout="wide"
)

# ---------------- ADVANCED CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

h1, h2, h3, h4 {
    color: #00f5d4;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141E30, #243B55);
}

.card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(15px);
    border-radius: 15px;
    padding: 25px;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("🛡 CrimeShield AI")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 About AI", "📊 Risk Analysis", "📂 Upload & Analyze"]
)

# =========================================================
# 🏠 PAGE 1 — ABOUT
# =========================================================

if page == "🏠 About AI":

    st.title("🛡 CrimeShield AI")
    st.subheader("Next-Generation Crime Intelligence System")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("""
### 🚀 What is CrimeShield AI?

CrimeShield AI is a machine learning-powered intelligence platform that:

✔ Predicts crime vulnerability  
✔ Detects geographic risk clusters  
✔ Generates AI-based explanation  
✔ Produces downloadable intelligence reports  
✔ Supports custom dataset upload  

---

### 🎯 Why It Matters

Smart cities require predictive AI systems to improve safety,
allocate police resources efficiently, and reduce crime proactively.

---

### 🛠 How To Use

1️⃣ Go to **Risk Analysis**  
2️⃣ View crime hotspot map  
3️⃣ Predict location risk  
4️⃣ Generate AI-powered report  
5️⃣ Upload your own dataset if needed  

""")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 📊 PAGE 2 — RISK ANALYSIS
# =========================================================

elif page == "📊 Risk Analysis":

    st.title("📊 Crime Risk Intelligence Dashboard")

    df = load_default_data()
    df = preprocess_data(df)

    st.success("Default Dataset Loaded")

    st.markdown("### 🗺 Crime Hotspot Map")

    fig = px.scatter_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        zoom=10,
        height=500
    )
    fig.update_layout(mapbox_style="carto-darkmatter")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## 🔍 Predict Risk")

    col1, col2 = st.columns(2)

    with col1:
        lat = st.number_input("Latitude", value=float(df["Latitude"].mean()))
        lon = st.number_input("Longitude", value=float(df["Longitude"].mean()))

    with col2:
        hour = st.slider("Hour", 0, 23, 12)
        weekend = st.selectbox("Weekend?", [0, 1])

    # -------- SESSION STATE FIX --------
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False

    if st.button("🚀 Analyze Location"):

        cvi = predict_risk(df, lat, lon, hour, weekend)

        st.session_state.analysis_done = True
        st.session_state.prediction = cvi
        st.session_state.cvi = cvi
        st.session_state.lat = lat
        st.session_state.lon = lon
        st.session_state.hour = hour
        st.session_state.weekend = weekend

    # -------- SHOW RESULTS --------
    if st.session_state.analysis_done:

        st.markdown("### 📈 Crime Vulnerability Index")
        st.metric("CVI Score", f"{cvi} / 100")

        if st.session_state.prediction == 1:
            st.error("⚠ HIGH RISK ZONE")
            explanation = """
This location shows elevated crime probability based on
historical geographic and time-based patterns.

Recommended Action:
• Increase surveillance
• Improve monitoring
• Deploy patrol units
"""
        else:
            st.success("✅ LOW RISK ZONE")
            explanation = """
This location shows lower crime probability.

Historical data indicates stable patterns.

Recommended Action:
• Maintain standard monitoring
"""

        st.markdown("### 🤖 AI Intelligence Explanation")
        st.info(explanation)

        # -------- PDF GENERATION --------
        report_text = f"""
CrimeShield AI Report

Location: {st.session_state.lat}, {st.session_state.lon}
Hour: {st.session_state.hour}
Weekend: {st.session_state.weekend}
CVI Score: {st.session_state.cvi}

AI Analysis:
{explanation}
"""

        file_path = generate_report(
        st.session_state.lat,
        st.session_state.lon,
        st.session_state.hour,
        st.session_state.weekend,
        st.session_state.cvi,
        explanation
)

        with open(file_path, "rb") as f:
            st.download_button(
                label="⬇ Download AI Report",
                data=f,
                file_name="CrimeShield_Report.pdf",
                mime="application/pdf"
            )

# =========================================================
# 📂 PAGE 3 — UPLOAD & ANALYZE
# =========================================================

elif page == "📂 Upload & Analyze":

    st.title("📂 Custom Dataset Intelligence")

    uploaded_file = st.file_uploader("Upload your CSV dataset")

    if uploaded_file:

        df = load_uploaded_data(uploaded_file)
        df = preprocess_data(df)

        st.success("Your Dataset Processed Successfully")

        st.markdown("### 🗺 Uploaded Data Hotspot Map")

        fig = px.scatter_mapbox(
            df,
            lat="Latitude",
            lon="Longitude",
            zoom=10,
            height=500
        )
        fig.update_layout(mapbox_style="carto-darkmatter")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📊 AI Insights")

        st.metric("Total Records", len(df))
        st.metric("Avg Latitude", round(df["Latitude"].mean(), 4))
        st.metric("Avg Longitude", round(df["Longitude"].mean(), 4))

        st.info("""
The AI automatically analyzes spatial clustering and
temporal patterns in your uploaded dataset.

For predictive scoring, use the Risk Analysis section.

""")



