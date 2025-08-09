import streamlit as st
import pandas as pd
import pickle

# 1. Load model & scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 2. Daftar kolom model (sesuaikan dengan fitur yang digunakan saat training!)
# Contoh: Jika model dilatih dengan fitur geografis, masukkan kolomnya di sini.
model_columns = [
    'Age', 'Family size', 'latitude', 'longitude',  # Kolom numerik
    'Gender_Female', 'Gender_Male',  # Kolom one-hot encoded
    'Marital Status_Married', 'Marital Status_Single',
    'Occupation_Employee', 'Occupation_Self-Employed', 'Occupation_Student',
    'Monthly Income_Below Rs.10000', 'Monthly Income_No Income',
    'Educational Qualifications_Bachelor', 'Educational Qualifications_High School',
    'Pin code_550001', 'Pin code_550010'  # Contoh kategori Pin code (sesuaikan dengan data Anda!)
]

st.title("🍔 Prediksi Customer Engagement - Online Food Service")

# 3. Form input (termasuk longitude, latitude, dan Pin code)
age = st.number_input("Age", min_value=15, max_value=80, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
occupation = st.selectbox("Occupation", ["Student", "Employee", "Self-Employed"])
income = st.selectbox("Monthly Income", ["No Income", "Below Rs.10000"])
education = st.selectbox("Educational Qualifications", ["High School", "Bachelor"])
family_size = st.number_input("Family size", min_value=1, max_value=10, value=3)
latitude = st.number_input("Latitude", format="%.6f", value=12.345678)
longitude = st.number_input("Longitude", format="%.6f", value=98.765432)
pin_code = st.selectbox("Pin code", ["550001", "550010", "550017"])  # Contoh pilihan

# 4. Tombol Prediksi
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

    # One-hot encoding
    input_encoded = pd.get_dummies(input_df)
    
    # Pastikan semua kolom training ada di input
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Urutkan kolom sesuai model
    input_encoded = input_encoded[model_columns]

    # Scaling dan prediksi
    input_scaled = scaler.transform(input_encoded)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]

    # Tampilkan hasil
    st.subheader("Hasil Prediksi")
    st.write(f"Output: **{'Yes' if pred == 1 else 'No'}**")
    st.write(f"Probabilitas: No ({prob[0]:.2f}), Yes ({prob[1]:.2f})")