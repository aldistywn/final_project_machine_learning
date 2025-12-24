import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sistem Prediksi Penyakit Jantung",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. FUNGSI LOAD MODEL (CACHED)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        # Memuat file model yang disimpan dari notebook
        return joblib.load('model_jantung.pkl')
    except FileNotFoundError:
        return None

# Load data
artifacts = load_artifacts()

# Cek apakah file berhasil di-load
if artifacts is None:
    st.error("❌ File 'model_jantung.pkl' tidak ditemukan!")
    st.warning("Silakan jalankan file 'analisis_model.ipynb' terlebih dahulu sampai selesai untuk membuat file model.")
    st.stop()

# Ekstrak komponen dari artifacts
model_xgb = artifacts['model_xgb']
model_dt = artifacts['model_dt']
preprocessor = artifacts['preprocessor']
feature_names = artifacts['feature_names']
X_train_sample = artifacts['X_train_sample']
comparison_metrics = artifacts.get('comparison_metrics', None) # Ambil metrics jika ada

# -----------------------------------------------------------------------------
# 3. SIDEBAR MENU
# -----------------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=100)
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Menu:", ["🏠 Home", "📝 Prediksi Pasien", "📊 Performa & Visualisasi"])

st.sidebar.divider()
st.sidebar.info("Dibuat Oleh: \n Ayudya Aldi Setiawan & Muhammad Bagus Zatrio Aji")

# -----------------------------------------------------------------------------
# 4. HALAMAN HOME
# -----------------------------------------------------------------------------
if menu == "🏠 Home":
    st.title("🫀 Sistem Deteksi Dini Penyakit Jantung")
    st.markdown("""
    Selamat datang di aplikasi prediksi risiko penyakit jantung. Aplikasi ini menggunakan **Machine Learning** untuk menganalisis data klinis pasien dan memberikan estimasi risiko.
    
    ### 🌟 Fitur Utama:
    * **Prediksi Cepat:** Menggunakan algoritma **XGBoost** yang telah dioptimasi.
    * **Analisis Komparatif:** Membandingkan performa dengan Decision Tree.
    * **Transparansi (Explainability):** Menjelaskan alasan di balik prediksi menggunakan **SHAP Values**.
    
    ### 📚 Tentang Model:
    Dataset yang digunakan berasal dari *UCI Machine Learning Repository* (920 sampel) yang telah melalui proses:
    1.  **Cleaning:** Penanganan data kosong & outlier.
    2.  **SMOTE:** Penyeimbangan data agar prediksi lebih adil.
    3.  **Tuning:** Menggunakan XGBoost Classifier sebagai model utama.
    """)
    
    st.info("👈 Silakan pilih menu **Prediksi Pasien** di sebelah kiri untuk memulai.")

# -----------------------------------------------------------------------------
# 5. HALAMAN PREDIKSI (INPUT DATA)
# -----------------------------------------------------------------------------
elif menu == "📝 Prediksi Pasien":
    st.header("📝 Form Data Klinis Pasien")
    st.write("Masukkan parameter kesehatan pasien di bawah ini:")
    
    # Form Input (Dibagi 2 Kolom agar rapi)
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Usia (Tahun)", 20, 100, 50)
        sex = st.selectbox("Jenis Kelamin", [1, 0], format_func=lambda x: "Laki-laki" if x==1 else "Perempuan")
        cp = st.selectbox("Tipe Nyeri Dada", [1, 2, 3, 4], 
                          format_func=lambda x: f"Tipe {x}: " + ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x-1])
        trestbps = st.number_input("Tekanan Darah (mm Hg)", 80, 200, 120)
        chol = st.number_input("Kolesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
        restecg = st.selectbox("Hasil EKG", [0, 1, 2], 
                               format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "LV Hypertrophy"][x])

    with col2:
        thalach = st.number_input("Detak Jantung Maksimum", 60, 220, 150)
        exang = st.selectbox("Nyeri Dada saat Olahraga?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
        slope = st.selectbox("Slope ST Segment", [1, 2, 3], 
                             format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x-1])
        ca = st.number_input("Jumlah Pembuluh Darah (0-3)", 0.0, 3.0, 0.0)
        thal = st.selectbox("Thalassemia", [3, 6, 7], 
                            format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversable Defect"}[x])

    # Tombol Eksekusi
    if st.button("🔍 Analisis Risiko Sekarang", type="primary"):
        # 1. Siapkan Data
        input_data = pd.DataFrame({
            'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
            'chol': [chol], 'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach],
            'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
        })
        
        try:
            # 2. Preprocessing
            input_processed = preprocessor.transform(input_data)
            
            # 3. Prediksi (Pakai XGBoost sebagai default terbaik)
            prediction = model_xgb.predict(input_processed)[0]
            probability = model_xgb.predict_proba(input_processed)[0][1]
            
            st.divider()
            
            # 4. Tampilkan Hasil
            col_hasil, col_shap = st.columns([1, 1])
            
            with col_hasil:
                st.subheader("Hasil Diagnosis Model")
                if prediction == 1:
                    st.error(f"⚠️ **BERISIKO TINGGI PENYAKIT JANTUNG**\n\nProbabilitas: **{probability:.1%}**")
                    st.markdown("Disarankan untuk segera melakukan pemeriksaan medis lanjut.")
                else:
                    st.success(f"✅ **KONDISI JANTUNG SEHAT/NORMAL**\n\nProbabilitas Sakit: **{probability:.1%}**")
                    st.markdown("Tetap jaga pola hidup sehat dan olahraga teratur.")

            # 5. Explainability (SHAP Local)
            with col_shap:
                st.subheader("Faktor Penentu")
                st.write("Fitur apa yang mendorong hasil prediksi ini?")
                
                # Hitung SHAP value untuk 1 data ini
                explainer = shap.TreeExplainer(model_xgb)
                shap_values_single = explainer.shap_values(input_processed)
                
                # Plot Bar Chart SHAP
                fig, ax = plt.subplots(figsize=(5, 4))
                shap.summary_plot(shap_values_single, input_processed, feature_names=feature_names, plot_type="bar", show=False)
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")

# -----------------------------------------------------------------------------
# 6. HALAMAN VISUALISASI (Global SHAP & Comparison)
# -----------------------------------------------------------------------------
elif menu == "📊 Performa & Visualisasi":
    st.header("📊 Analisis Performa Model & Visualisasi")
    
    # Ambil data matrix dari artifacts
    cms = artifacts.get('confusion_matrices', None)

    # TAB 1: PERBANDINGAN MODEL
    st.subheader("1. Perbandingan: XGBoost vs Decision Tree")
    
    if comparison_metrics is not None:
        # Tampilkan Dataframe Metrik
        st.dataframe(comparison_metrics.style.format({
            'Akurasi': '{:.1%}',
            'Recall': '{:.1%}',
            'Precision': '{:.1%}',
            'F1-Score': '{:.1%}'
        }), use_container_width=True)
        
        # --- BAGIAN BARU: CONFUSION MATRIX ---
        if cms is not None:
            st.write("#### Confusion Matrix")
            st.write("Visualisasi detail tebakan benar vs salah:")
            
            col_cm1, col_cm2 = st.columns(2)
            
            # Helper function untuk plot heatmap
            def plot_cm(cm, title):
                fig, ax = plt.subplots(figsize=(3, 2.5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=['Sehat', 'Sakit'], 
                            yticklabels=['Sehat', 'Sakit'])
                ax.set_title(title, fontsize=10)
                ax.set_xlabel('Prediksi Model', fontsize=8)
                ax.set_ylabel('Kondisi Asli', fontsize=8)
                return fig

            with col_cm1:
                st.write("**Decision Tree**")
                fig_dt = plot_cm(cms['Decision Tree'], "Matrix Decision Tree")
                st.pyplot(fig_dt)
                
            with col_cm2:
                st.write("**XGBoost (Recommended)**")
                fig_xgb = plot_cm(cms['XGBoost'], "Matrix XGBoost")
                st.pyplot(fig_xgb)
            
            st.info("""
            **Cara Membaca Confusion Matrix:**
            * ↘️ **Diagonal Utama (Biru Gelap):** Jumlah tebakan yang **BENAR**.
            * ↗️ **Pojok Kanan Atas (False Positive):** Orang Sehat ditebak Sakit.
            * ↙️ **Pojok Kiri Bawah (False Negative):** Orang Sakit ditebak Sehat (**Paling Berbahaya di Medis!**).
            """)
        # -------------------------------------

        # Grafik Batang (Altair) - Lanjutan kode lama
        st.write("#### Grafik Perbandingan Metrik")
        df_melt = comparison_metrics.melt(id_vars="Model", var_name="Metrik", value_name="Nilai")
        
        chart = alt.Chart(df_melt).mark_bar().encode(
            x=alt.X('Metrik', axis=None, title=None),
            y=alt.Y('Nilai', axis=alt.Axis(format='%'), title='Skor'),
            color=alt.Color('Metrik', legend=alt.Legend(title="Jenis Metrik")),
            column=alt.Column('Model', header=alt.Header(titleOrient="bottom", labelFontSize=12))
        ).properties(width=150, height=300)
        
        st.altair_chart(chart, use_container_width=False)
        
    else:
        st.warning("⚠️ Data perbandingan model belum tersedia. Pastikan Anda sudah menjalankan ulang notebook 'analisis_model.ipynb' bagian paling bawah.")

    st.divider()

    # TAB 2: GLOBAL SHAP (Tetap sama seperti sebelumnya)
    st.subheader("2. Interpretasi Model Global (SHAP)")
    # ... (lanjutan kode SHAP sama seperti sebelumnya) ...
    if st.button("Tampilkan Plot SHAP Global"):
        with st.spinner("Sedang membuat plot..."):
            explainer = shap.TreeExplainer(model_xgb)
            shap_values = explainer.shap_values(X_train_sample)
            
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_train_sample, feature_names=feature_names, show=False)
            st.pyplot(fig)