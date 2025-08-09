import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder

# 1. Load Model & Scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("encoder.pkl", "rb") as f:  # File encoder yang disimpan saat training
    encoder = pickle.load(f)

# 2. Daftar fitur KATEGORI yang digunakan saat training (SESUAIKAN!)
kategori_features = {
    'Gender': ['Male', 'Female'],
    'Marital Status': ['Single', 'Married'],
    'Occupation': ['Student', 'Employee', 'Self-Employed'],
    'Monthly Income': ['No Income', 'Below Rs.10000'],
    'Educational Qualifications': ['High School', 'Bachelor', 'Master', 'PhD'],
    'Pin code': ['550001', '550010', '550017']
}

st.title("🍔 Prediksi Customer Engagement")

# 3. Form Input
with st.form("input_form"):
    age = st.number_input("Usia", min_value=15, max_value=80, value=25)
    gender = st.selectbox("Jenis Kelamin", kategori_features['Gender'])
    marital_status = st.selectbox("Status Perkawinan", kategori_features['Marital Status'])
    occupation = st.selectbox("Pekerjaan", kategori_features['Occupation'])
    income = st.selectbox("Pendapatan Bulanan", kategori_features['Monthly Income'])
    education = st.selectbox("Kualifikasi Pendidikan", kategori_features['Educational Qualifications'])
    family_size = st.number_input("Jumlah Keluarga", min_value=1, max_value=10, value=3)
    latitude = st.number_input("Latitude", format="%.6f", value=12.345678)
    longitude = st.number_input("Longitude", format="%.6f", value=98.765432)
    pin_code = st.selectbox("Kode Pos", kategori_features['Pin code'])
    
    submitted = st.form_submit_button("Prediksi")

if submitted:
    try:
        # 4. Buat DataFrame Input
        input_data = {
            'Age': [age],
            'Gender': [gender],
            'Marital Status': [marital_status],
            'Occupation': [occupation],
            'Monthly Income': [income],
            'Educational Qualifications': [education],
            'Family size': [family_size],
            'latitude': [latitude],
            'longitude': [longitude],
            'Pin code': [pin_code]
        }
        input_df = pd.DataFrame(input_data)

        # 5. Encoding Kategori dengan OneHotEncoder yang SAMA seperti training
        encoded_cat = encoder.transform(input_df[['Gender', 'Marital Status', 'Occupation', 
                                               'Monthly Income', 'Educational Qualifications', 
                                               'Pin code']]).toarray()
        
        # 6. Gabungkan dengan fitur numerik
        numeric_data = input_df[['Age', 'Family size', 'latitude', 'longitude']].values
        final_input = np.concatenate([numeric_data, encoded_cat], axis=1)

        # 7. Scaling & Prediksi
        input_scaled = scaler.transform(final_input)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0]

        # 8. Tampilkan Hasil
        st.success(f"Hasil: {'Ya' if pred == 1 else 'Tidak'}")
        st.write(f"Probabilitas: Tidak ({prob[0]:.2%}), Ya ({prob[1]:.2%})")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("**Debug Info:**")
        st.write("Input:", input_df)