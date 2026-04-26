import streamlit as st
import pandas as pd
import joblib

# Load models and preprocessors
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# --- App Title ---
st.markdown("<h1 style='text-align: center; color: red;'>❤️ Heart Stroke Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>by Anuj</h4>", unsafe_allow_html=True)
st.write("---")

# --- Sidebar Info ---
st.sidebar.header("ℹ️ About")
st.sidebar.info(
    "This app predicts the **risk of heart disease** "
    "based on patient medical details using a KNN model."
)

# --- User Inputs ---
st.subheader("📋 Provide Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# --- Prediction Button ---
st.write("---")
if st.button("🔍 Predict Risk"):
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "chestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([raw_input])

    # Ensure all expected columns exist
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    # Scale and Predict
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    # --- Display Result ---
    st.write("---")
    if prediction == 1:
        st.error("🚨 **High Risk of Heart Disease Detected!** Consult a doctor immediately.")
    else:
        st.success("✅ **Low Risk of Heart Disease.** Stay healthy!")

