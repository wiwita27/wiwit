import streamlit as st
import pandas as pd
import pickle

# 1. Load model & scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🍔 Prediksi Customer Engagement - Online Food Service")

# 2. Form input sesuai kolom dataset (sebelum one-hot encoding)
age = st.number_input("Age", min_value=15, max_value=80, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
occupation = st.selectbox("Occupation", ["Student", "Employee", "Self-Employed", "Unemployed"])
income = st.selectbox("Monthly Income", ["Low", "Medium", "High"])
education = st.selectbox("Educational Qualifications", ["High School", "Bachelor", "Master", "PhD"])
family_size = st.number_input("Family size", min_value=1, max_value=10, value=3)
latitude = st.number_input("Latitude", format="%.6f")
longitude = st.number_input("Longitude", format="%.6f")
pin_code = st.text_input("Pin code")

# 3. Saat tombol diklik
if st.button("Prediksi"):
    # Buat DataFrame dari input user
    input_df = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Marital Status": [marital_status],
        "Occupation": [occupation],
        "Monthly Income": [income],
        "Educational Qualifications": [education],
        "Family size": [family_size],
        "latitude": [latitude],
        "longitude": [longitude],
        "Pin code": [pin_code]
    })

    # One-hot encoding (harus sama seperti saat training)
    input_encoded = pd.get_dummies(input_df)
    
    # Baca semua kolom yang digunakan model saat training
    # dan isi kolom yang hilang dengan 0
    model_columns = model.feature_names_in_
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_columns]

    # Scaling
    input_scaled = scaler.transform(input_encoded)

    # Prediksi
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]

    st.subheader("Hasil Prediksi")
    st.write(f"Output: **{pred}**")
    st.write(f"Probabilitas: {prob}")
