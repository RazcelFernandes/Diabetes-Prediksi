from fpdf import FPDF
import numpy as np
import streamlit as st
import joblib
import os
import plotly.graph_objects as go

# Set page config harus dipanggil pertama kali
st.set_page_config(page_title="Deteksi Diabetes", layout="centered")

# CSS yang diperbaiki untuk tampilan yang lebih menarik
st.markdown("""
    <style>
    /* Styling dasar aplikasi */
    .stApp {
        background-image: url('https://i.pinimg.com/originals/9f/f7/d1/9ff7d1a690e7bf508eda106f9bc13dab.jpg');
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        transition: all 0.3s ease-in-out;
    }
    
    /* Overlay dan animasi */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.85);
        z-index: -1;
    }

    .main {
        animation: fadeIn 0.5s ease-in-out;
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Styling komponen */
    .css-1d391kg {
        padding: 2rem;
        border-radius: 1rem;
        background-color: rgba(255, 255, 255, 0.9);
        transition: all 0.3s ease;
    }
    
    .stTitle {
        color: #2c3e50;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .stSubheader {
        color: #34495e;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Form styling */
    div[data-testid="stForm"] {
        border: 1px solid #e0e0e0;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        background-color: white;
        transition: all 0.3s ease;
        animation: slideIn 0.5s ease-in-out;
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Input dan text styling */
    div.stNumberInput label {
        color: #2c3e50 !important;
        font-weight: 500 !important;
    }

    div.stAlert > div {
        color: #2c3e50 !important;
    }

    div.stMarkdown p {
        color: #2c3e50 !important;
    }

    .js-plotly-plot .plotly .gtitle {
        color: #2c3e50 !important;
    }

    /* Card animations */
    div[class*="stMarkdown"] > div {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    div[class*="stMarkdown"] > div:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Menambahkan emoji dan styling ke judul
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>🩺 Aplikasi Prediksi Diabetes</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic; color: #7f8c8d;'>Masukkan data pasien untuk memprediksi risiko diabetes</p>", unsafe_allow_html=True)

# Load model dan scaler
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'models', 'diabetes_model.pkl')
scaler_path = os.path.join(current_dir, '..', 'models', 'scaler.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.error(f"Gagal memuat model: {str(e)}")
    st.stop()

# Membuat sidebar untuk navigasi
st.sidebar.title("Menu")
menu = st.sidebar.radio("Pilih Menu:", ["🏠 Home", "👤 Data Pasien", "📝 Input Data"])

if menu == "🏠 Home":
    
    # Menambahkan informasi tentang aplikasi
    st.markdown("""
    <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        <h2 style='color: #2c3e50;'>👋 Tentang Aplikasi</h2>
        <p style='color: #34495e;'>Aplikasi ini dirancang untuk membantu mendeteksi risiko diabetes berdasarkan parameter kesehatan pasien. Dengan menggunakan machine learning, aplikasi ini dapat memberikan prediksi awal tentang kemungkinan seseorang mengidap diabetes.</p>
    </div>
    """, unsafe_allow_html=True)

    # Menambahkan fitur utama
    st.markdown("""
    <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 20px;'>
        <h3 style='color: #2c3e50;'>✨ Fitur Utama</h3>
        <ul style='color: #34495e;'>
            <li>Prediksi risiko diabetes berdasarkan data medis</li>
            <li>Visualisasi hasil prediksi yang mudah dipahami</li>
            <li>Tampilan yang user-friendly</li>
            <li>Hasil prediksi instan</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Menambahkan parameter yang digunakan
    st.markdown("""
    <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 20px;'>
        <h3 style='color: #2c3e50;'>📋 Parameter Prediksi</h3>
        <ul style='color: #34495e;'>
            <li>🤰 Jumlah Kehamilan (untuk pasien wanita)</li>
            <li>🩸 Kadar Glukosa</li>
            <li>💉 Tekanan Darah</li>
            <li>📏 Ketebalan Kulit</li>
            <li>💊 Insulin</li>
            <li>⚖️ BMI (Berat Badan : Tinggi badan x 2 = contoh : 70 BB : 1.75 TB X2 = 22.86  )</li>
            <li>👪 Diabetes Pedigree Function</li>
            <li>📅 Usia</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Menambahkan cara penggunaan
    st.markdown("""
    <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 20px;'>
        <h3 style='color: #2c3e50;'>🔍 Cara Penggunaan</h3>
        <ol style='color: #34495e;'>
            <li>Pilih menu "Input Data" di sidebar</li>
            <li>Masukkan data sesuai parameter yang diminta</li>
            <li>Klik tombol "Prediksi" untuk melihat hasil</li>
            <li>Sistem akan menampilkan hasil prediksi beserta probabilitasnya</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # Menambahkan catatan penting
    st.markdown("""
    <div style='background-color: #fff3e0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 20px;'>
        <h3 style='color: #e65100;'>⚠️ Catatan Penting</h3>
        <p style='color: #795548;'>Aplikasi ini dirancang untuk membantu menentukan apakah Anda berisiko terkena penyakit diabetes. Mohon masukkan data yang akurat dan benar.
        <p> Aplikasi ini di rancang oleh Razcel Fernandes.</p>
    </div>
    """, unsafe_allow_html=True)

elif menu == "👤 Data Pasien":
    st.markdown("<h2 style='text-align: center; color: #2c3e50;'>👤 Data Pribadi Pasien</h2>", unsafe_allow_html=True)
    
    # Menggunakan session state untuk menyimpan data pasien
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = {
            'nama': '',
            'gender': '',
            'alamat': '',
            'riwayat': ''
        }
    
    with st.form("form_patient_data"):
        # Menggunakan CSS untuk mengubah warna teks menjadi hitam
        st.markdown("""
            <style>
            div[data-testid="stForm"] label {
                color: black !important;
            }
            div[data-testid="stForm"] .stSelectbox label {
                color: black !important;
            }
            div[data-testid="stForm"] .stTextArea label {
                color: black !important;
            }
            div[data-testid="stForm"] .stTextInput label {
                color: black !important;
            }
            .stMarkdown {
                color: black !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Input nama pasien
        nama = st.text_input('👤 Nama Lengkap Pasien', value=st.session_state.patient_data['nama'])
        
        # Input jenis kelamin
        gender = st.selectbox('⚧ Jenis Kelamin', 
                            options=['Pilih Jenis Kelamin', 'Laki-laki', 'Perempuan'],
                            index=0 if not st.session_state.patient_data['gender'] else 
                            ['Pilih Jenis Kelamin', 'Laki-laki', 'Perempuan'].index(st.session_state.patient_data['gender']))
        
        # Input alamat
        alamat = st.text_area('🏠 Alamat Lengkap', value=st.session_state.patient_data['alamat'])
        
        # Input riwayat penyakit
        riwayat = st.text_area('📋 Riwayat Penyakit (jika ada)', 
                              help="Masukkan riwayat penyakit yang pernah diderita",
                              value=st.session_state.patient_data['riwayat'])
        
        submitted = st.form_submit_button("💾 Simpan Data")
        
        if submitted:
            if not nama or gender == 'Pilih Jenis Kelamin' or not alamat:
                st.error("⚠️ Mohon lengkapi data nama, jenis kelamin, dan alamat!")
            else:
                st.session_state.patient_data = {
                    'nama': nama,
                    'gender': gender,
                    'alamat': alamat,
                    'riwayat': riwayat
                }
                st.success("✅ Data pasien berhasil disimpan!")
                st.info("ℹ️ Silahkan lanjut ke menu 'Input Data' untuk melakukan prediksi diabetes")

elif menu == "📝 Input Data":
    # Cek apakah data pasien sudah diisi
    if 'patient_data' not in st.session_state or not st.session_state.patient_data['nama']:
        st.warning("⚠️ Mohon isi data pribadi pasien terlebih dahulu di menu 'Data Pasien'")
        st.stop()
    
    # Menampilkan info pasien
    st.markdown(f"""
    <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        <h3 style='color: #2c3e50;'>👤 Informasi Pasien</h3>
        <p><strong>Nama:</strong> {st.session_state.patient_data['nama']}</p>
        <p><strong>Jenis Kelamin:</strong> {st.session_state.patient_data['gender']}</p>
        <p><strong>Alamat:</strong> {st.session_state.patient_data['alamat']}</p>
        <p><strong>Riwayat Penyakit:</strong> {st.session_state.patient_data['riwayat'] or '-'}</p>
    </div>
    """, unsafe_allow_html=True)

    # Form Input Data dengan styling yang lebih baik
    with st.form("form_diabetes"):
        st.markdown("<h3 style='color: #2c3e50;'>📋 Data Medis Pasien</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input('🤰 Pregnancies (Untuk wanita Hamil)', min_value=0.0, value=0.0)
            glucose = st.number_input('🩸 Glucosa', min_value=0.0, value=0.0)
            blood_pressure = st.number_input('💉 BloodPressure', min_value=0.0, value=0.0)
            skin_thickness = st.number_input('📏 SkinThickness ', min_value=0.0, value=0.0)

        with col2:
            insulin = st.number_input('💊 Insulin', min_value=0.0, value=0.0)
            bmi = st.number_input('⚖️ BMI', min_value=0.0, value=0.0)
            dpf = st.number_input('👪 Diabetes Pedigree Function', min_value=0.0, value=0.0)
            age = st.number_input('📅 Age', min_value=0.0, value=0.0)

        submitted = st.form_submit_button("🔍 Prediksi")

        # Prediksi dengan styling yang lebih menarik
        if submitted:
            input_data = [pregnancies, glucose, blood_pressure, skin_thickness,
                        insulin, bmi, dpf, age]
            input_array = np.array([input_data], dtype=np.float64)

            # Mengubah validasi agar Pregnancies bisa 0
            if np.isnan(input_array).any() or any(
                value == 0.0 for i, value in enumerate(input_data) if i != 0  # Mengizinkan Pregnancies = 0
            ):
                st.error("⚠️ Terdapat nilai kosong atau tidak valid pada input! (kecuali data kehamilan)")
            else:
                try:
                    input_scaled = scaler.transform(input_array)
                    prediction = model.predict(input_scaled)
                    proba = model.predict_proba(input_scaled)

                    # Hasil prediksi dengan styling yang lebih menarik
                    st.markdown("<h3 style='color: #2c3e50; text-align: center;'>📊 Hasil Prediksi</h3>", unsafe_allow_html=True)
                    left_col, right_col = st.columns(2)

                    with left_col:
                        st.markdown("<div style='margin-top: 59px;'>", unsafe_allow_html=True)  # Menambahkan margin atas
                        result = '🔴 Positif Diabetes' if prediction[0] == 1 else '🟢 Negatif Diabetes'
                        st.markdown(f"<div style='background-color: {'#ffebee' if prediction[0] == 1 else '#e8f5e9'}; padding: 1rem; border-radius: 10px; text-align: center;'><h4 style='color: #000000;'>{result}</h4></div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='background-color: #e3f2fd; padding: 1rem; border-radius: 10px; text-align: center; margin-top: 10px; border: 1px solid #90caf9;'><p style='color: #1565c0; font-weight: bold; margin: 0;'>📊 Probabilitas: {proba[0][1]*100:.2f}%</p></div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    with right_col:
                        # Membuat diagram gauge (indikator) dengan warna hijau, kuning, merah
                        gauge_value = proba[0][1] * 100
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=gauge_value,
                            title={'text': "Probabilitas Diabetes (%)", 'font': {'size': 16}},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'steps': [
                                    {'range': [0, 40], 'color': "lightgreen"},   # Aman
                                    {'range': [40, 70], 'color': "yellow"},       # Waspada
                                    {'range': [70, 100], 'color': "tomato"}       # Risiko tinggi
                                ],
                                'bar': {'color': "darkred"}  # Warna jarum indikator
                            },
                            number={'font': {'size': 30, 'color': "black"}}
                        ))
                        fig.update_layout(
                            width=350, 
                            height=250,  # Mengurangi tinggi diagram
                            margin=dict(t=30, b=0, l=30, r=30),  # Mengatur margin untuk posisi yang lebih baik
                            paper_bgcolor='rgba(0,0,0,0)',  
                            plot_bgcolor='rgba(0,0,0,0)'    
                        )
                        st.plotly_chart(fig)
                        

                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan: {str(e)}")
