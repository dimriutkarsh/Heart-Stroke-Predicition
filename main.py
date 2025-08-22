import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing tools
model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

# Page config
st.set_page_config(page_title="Heart Stroke Prediction", page_icon="❤️", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
        .title {
            font-size:40px !important;
            color:#C0392B;
            text-align:center;
            font-weight:bold;
            margin-bottom:20px;
        }
        .subtitle {
            font-size:18px;
            text-align:center;
            color:#34495E;
            margin-bottom:30px;
        }
        .result-box {
            padding:20px;
            border-radius:12px;
            text-align:center;
            font-size:22px;
            font-weight:bold;
            margin-top:20px;
        }
        .success {
            background-color:#D5F5E3;
            color:#196F3D;
            border:2px solid #27AE60;
        }
        .danger {
            background-color:#FADBD8;
            color:#922B21;
            border:2px solid #C0392B;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<div class="title">❤️ Heart Stroke Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Provide the following details to check your risk</div>', unsafe_allow_html=True)

# Input fields (line by line)
age = st.slider("🎂 Age", 18, 100, 40)
sex = st.selectbox("🧑‍⚕️ Sex", ['M', 'F'])
chestpain = st.selectbox("💔 Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("🩸 Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("🥓 Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("📈 Resting ECG Results", ["Normal", "ST", "LVH"])
max_hr = st.slider("💓 Maximum Heart Rate Achieved", 60, 220, 150)   # ✅ fixed
exercise_angina = st.selectbox("🏃 Exercise Induced Angina", ['Y', 'N'])
oldpeak = st.number_input("📉 Oldpeak (ST depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("⛰️ ST Slope", ["Up", "Flat", "Down"])

# Prediction
if st.button("🔍 Predict"):
    raw_input = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chestpain,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    # Styled result
    if prediction == 1:
        st.markdown('<div class="result-box danger">🚨 High Risk of Heart Disease! Please consult a doctor immediately.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box success">✅ Low Risk of Heart Disease! Keep maintaining a healthy lifestyle.</div>', unsafe_allow_html=True)
