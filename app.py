import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load Model & Scaler
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Gagal memuat model: {str(e)}")
    st.stop()

# 2. Daftar Kolom Model (SESUAIKAN DENGAN MODEL ANDA!)
model_columns = [
    'Age', 'Family size', 'latitude', 'longitude',
    'Gender_Female', 'Gender_Male',
    'Marital Status_Married', 'Marital Status_Single',
    'Occupation_Employee', 'Occupation_Self-Employed', 'Occupation_Student',
    'Monthly Income_Below Rs.10000', 'Monthly Income_No Income',
    'Educational Qualifications_Bachelor', 'Educational Qualifications_High School',
    'Pin code_550001', 'Pin code_550010'  # Contoh kategori Pin code
]

st.title("🍔 Prediksi Customer Engagement")

# 3. Form Input
with st.form("input_form"):
    age = st.number_input("Age", min_value=15, max_value=80, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    occupation = st.selectbox("Occupation", ["Student", "Employee", "Self-Employed"])
    income = st.selectbox("Monthly Income", ["No Income", "Below Rs.10000"])
    education = st.selectbox("Educational Qualifications", ["High School", "Bachelor"])
    family_size = st.number_input("Family size", min_value=1, max_value=10, value=3)
    latitude = st.number_input("Latitude", format="%.6f", value=12.345678)
    longitude = st.number_input("Longitude", format="%.6f", value=98.765432)
    pin_code = st.selectbox("Pin code", ["550001", "550010", "550017"])
    
    submitted = st.form_submit_button("Prediksi")

if submitted:
    try:
        # 4. Buat DataFrame Input
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

        # 5. One-Hot Encoding
        input_encoded = pd.get_dummies(input_df)
        
        # 6. Pastikan Semua Kolom Model Ada
        for col in model_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # 7. Urutkan Kolom & Validasi
        input_encoded = input_encoded[model_columns]
        input_encoded = input_encoded.astype(float)  # Pastikan tipe data float

        # 8. Scaling & Prediksi
        input_scaled = scaler.transform(input_encoded)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0]

        # 9. Tampilkan Hasil
        st.success(f"Hasil Prediksi: {'Ya' if pred == 1 else 'Tidak'}")
        st.write(f"Probabilitas: Tidak ({prob[0]:.2%}), Ya ({prob[1]:.2%})")

    except Exception as e:
        st.error(f"Terjadi error: {str(e)}")
        st.write("*Debug Info:*")
        st.write("Kolom Input:", input_encoded.columns.tolist())
        st.write("Kolom Model:", model_columns)