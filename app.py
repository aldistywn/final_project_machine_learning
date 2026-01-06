import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import time
from datetime import datetime
from fpdf import FPDF

# ==========================================
# 1. CONFIG & CSS STYLING
# ==========================================
st.set_page_config(
    page_title="CardioCheck | Prediksi Jantung",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS agar tampilan profesional
st.markdown("""
    <style>
    .reportview-container {background: #f0f2f6}
    .main-title {font-size: 3rem; color: #D32F2F; text-align: center; font-weight: 800; margin-bottom: 10px;}
    .sub-title {text-align: center; color: #555; margin-bottom: 30px;}
    div.stButton > button:first-child {background-color: #D32F2F; color: white;}
    
    /* Tambahan CSS untuk Zona Risiko */
    .risk-mid {padding:15px; border-radius:10px; background-color:#fff3cd; color:#856404; border: 1px solid #ffeeba; text-align: center;}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI PEMBUAT PDF
# ==========================================
def create_pdf(patient_name, patient_data, result_label, proba, alerts):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Prediksi Penyakit Jantung - Laporan Medis", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, "Sistem Deteksi Dini Penyakit Jantung Berbasis AI", ln=True, align='C')
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)
    
    # Tanggal
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Tanggal Pemeriksaan: {datetime.now().strftime('%d-%m-%Y %H:%M')}", ln=True)
    pdf.ln(5)
    
    # 1. Data Pasien
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 10, "1. Identitas & Klinis Pasien", ln=True, fill=True)
    pdf.set_font("Arial", '', 12)
    
    # --- MENAMPILKAN NAMA (BARU) ---
    pdf.cell(50, 8, "Nama Pasien", border=0)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, f": {patient_name.upper()}", ln=True)
    pdf.set_font("Arial", '', 12)
    
    # Loop sisa data
    for key, value in patient_data.items():
        pdf.cell(50, 8, f"{key}", border=0)
        pdf.cell(0, 8, f": {value}", ln=True)
        
    pdf.ln(5)
    
    # 2. Hasil Diagnosa
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. Hasil Analisis AI", ln=True, fill=True)
    pdf.set_font("Arial", '', 12)
    
    pdf.cell(50, 8, "Kesimpulan", border=0)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, f": {result_label}", ln=True)
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(50, 8, "Probabilitas Risiko", border=0)
    pdf.cell(0, 8, f": {proba*100:.2f}%", ln=True)
    
    # 3. Catatan Khusus
    if alerts:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(200, 0, 0) # Merah
        pdf.cell(0, 10, "3. Peringatan Tanda Vital (Safety Net)", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.set_text_color(0, 0, 0) # Hitam
        for alert in alerts:
            clean_alert = alert.replace('**', '').replace('🔴', '').replace('⚠️', '')
            pdf.cell(0, 8, f"- {clean_alert.strip()}", ln=True)
            
    # Disclaimer
    pdf.set_y(-30)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Dokumen ini dihasilkan oleh komputer (CDSS). Validasi dokter tetap diperlukan.", ln=True, align='C')
    
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 3. LOAD DATA & MODEL
# ==========================================
@st.cache_resource
def load_helper():
    try:
        return joblib.load('model_jantung.pkl')
    except FileNotFoundError:
        return None

data_artifacts = load_helper()

if not data_artifacts:
    st.error("⚠️ File model 'model_jantung.pkl' hilang. Harap jalankan notebook training dulu.")
    st.stop()

# Load komponen dari artifacts
model = data_artifacts['model_xgb'] 
preprocessor = data_artifacts['preprocessor']
feat_names = data_artifacts['feature_names']
X_sample = data_artifacts['X_train_sample']
metrics = data_artifacts.get('comparison_metrics', None)
cms = data_artifacts.get('confusion_matrices', None)

# ==========================================
# 4. SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=80)
    st.title("Prediksi Penyakit Jantung")
    st.markdown("---")
    menu = st.radio("Main Menu", ["Dashboard", "Diagnosa Pasien", "Model Performance"])

# ==========================================
# 5. HALAMAN DASHBOARD
# ==========================================
if menu == "Dashboard":
    st.markdown('<div class="main-title">Sistem Prediksi Penyakit Jantung</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Sistem Prediksi Dini Penyakit Jantung Berbasis XGBoost</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("👋 Selamat Datang, Dokter.")
        st.write("""
        Sistem ini dirancang untuk membantu tenaga medis melakukan screening risiko penyakit jantung 
        menggunakan algoritma **XGBoost**. 
        
        **Keunggulan Model Ini:**
        1.  **Medical-Grade Cleaning:** Data dilatih dengan penanganan nilai medis yang valid.
        2.  **Soft Outlier Removal:** Menggunakan IQR 2.0 untuk mempertahankan data kasus sulit.
        3.  **Explainable AI:** Dilengkapi dengan analisis SHAP untuk transparansi keputusan.
        """)

    with col2:
        st.warning("⚠️ **Disclaimer**")
        st.caption("""
        Aplikasi ini adalah alat bantu keputusan (CDSS) dan bukan pengganti diagnosa klinis. 
        Keputusan akhir tetap berada di tangan tenaga medis profesional.
        """)

# ==========================================
# 6. HALAMAN DIAGNOSA (INPUT)
# ==========================================
elif menu == "Diagnosa Pasien":
    st.subheader("📋 Input Parameter Klinis")
    
    # Tombol Demo Data
    if st.button("Isi Data Contoh (Demo Pasien Sakit)"):
        st.session_state_defaults = {'name': 'Budi Santoso', 'age': 58, 'trestbps': 150, 'chol': 240, 'thalach': 110, 'oldpeak': 2.5, 'cp': 4}
    else:
        st.session_state_defaults = {}

    with st.form("medical_form"):
        st.markdown("**1. Identitas Pasien**")
        name = st.text_input("Nama Lengkap", st.session_state_defaults.get('name', ''), placeholder="Masukkan nama pasien...")

        # Group 1: Tanda Vital
        st.markdown("**1. Profil & Tanda Vital**")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Usia", 20, 100, st.session_state_defaults.get('age', 45))
            sex = st.selectbox("Jenis Kelamin", [1, 0], format_func=lambda x: "Laki-laki" if x==1 else "Perempuan")
        with c2:
            trestbps = st.number_input("Tekanan Darah (mmHg)", 80, 180, st.session_state_defaults.get('trestbps', 120))
            chol = st.number_input("Kolesterol (mg/dl)", 100, 600, st.session_state_defaults.get('chol', 200))
        with c3:
            fbs = st.selectbox("Gula Darah > 120 mg/dl?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")

        st.markdown("---")
        
        # Group 2: Jantung
        st.markdown("**2. Pemeriksaan Jantung**")
        c4, c5 = st.columns(2)
        with c4:
            cp = st.selectbox("Tipe Nyeri Dada", [1, 2, 3, 4], 
                              format_func=lambda x: f"Tipe {x}: " + ["Typical", "Atypical", "Non-anginal", "Asymptomatic"][x-1])
            restecg = st.selectbox("Hasil EKG", [0, 1, 2], format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
            thalach = st.number_input("Detak Jantung Max", 60, 250, st.session_state_defaults.get('thalach', 150))
            exang = st.selectbox("Nyeri Dada (Olahraga)", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
        
        with c5:
            oldpeak = st.number_input("ST Depression", 0.0, 10.0, st.session_state_defaults.get('oldpeak', 0.0))
            slope = st.selectbox("Slope ST", [1, 2, 3], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x-1])
            ca = st.number_input("Jml Pembuluh Darah (0-3)", 0.0, 3.0, st.session_state_defaults.get('ca', 0.0))
            thal = st.selectbox("Thalassemia", [3, 6, 7], format_func=lambda x: {3:"Normal", 6:"Fixed Defect", 7:"Reversable"}[x])

        submit = st.form_submit_button("🔍 Jalankan Analisis")

    # LOGIC PREDIKSI (YANG DIMODIFIKASI)
    if submit:
        with st.spinner("Sedang memproses data..."):
            time.sleep(0.5)

            # Validasi Nama
            if not name.strip():
                st.warning("⚠️ Mohon isi **Nama Pasien** terlebih dahulu.")
                st.stop()
            
            # Input DataFrame
            input_df = pd.DataFrame([{
                'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 
                'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang, 
                'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
            }])
            
            try:
                # Preprocessing & Prediksi
                X_proc = preprocessor.transform(input_df)
                pred_proba = model.predict_proba(X_proc)[0][1]

                st.divider()
                col_res, col_shap = st.columns([1, 1.5])

                # Variabel untuk PDF
                final_result_text = ""
                alerts = []
                
                with col_res:
                    st.subheader("Hasil Diagnosis")

                    st.subheader(f"Hasil Diagnosis: {name}")
                    
                    # --- LOGIKA 3 WARNA (MERAH, KUNING, HIJAU) ---
                    if pred_proba > 0.50:
                        # Zona MERAH (Risiko Tinggi)
                        final_result_text = "RISIKO TINGGI (Positif Penjakit Jantung)"
                        st.error(f"⚠️ **{final_result_text}**")
                        st.metric("Probabilitas Risiko", f"{pred_proba*100:.2f}%")
                        st.progress(int(pred_proba*100))
                        st.markdown("Saran: Segera lakukan pemeriksaan Angiografi.")
                        
                    elif pred_proba > 0.20: 
                        # Zona KUNING (Risiko Menengah / Waspada)
                        final_result_text = "RISIKO MENENGAH (Waspada)"
                        st.warning(f"⚠️ **{final_result_text}**")
                        st.metric("Probabilitas Risiko", f"{pred_proba*100:.2f}%")
                        st.progress(int(pred_proba*100))
                        st.markdown("Saran: Segera lakukan pemeriksaan Angiografi.")
                        
                    else:
                        # Zona HIJAU (Aman)
                        final_result_text = "RISIKO RENDAH (Aman)"
                        st.success(f"✅ **{final_result_text}**")
                        st.metric("Probabilitas Risiko", f"{pred_proba*100:.2f}%")
                        st.progress(int(pred_proba*100))
                        st.markdown("Saran: Pertahankan pola hidup sehat.")

                    # --- PERINGATAN TANDA VITAL (SAFETY NET) ---
                    # Ini akan muncul APAPUN hasil prediksinya
                    alerts = []
                    if trestbps >= 140:
                        alerts.append(f"🔴 **Tekanan Darah Tinggi ({trestbps} mmHg)**")
                    if chol >= 240:
                        alerts.append(f"🔴 **Kolesterol Tinggi ({chol} mg/dl)**")
                    
                    if alerts:
                        st.markdown("---")
                        st.warning("🚨 **PERHATIAN MEDIS**")
                        st.caption("Tanda vital berikut memerlukan penanganan:")
                        for a in alerts:
                            st.write(a)
                    
                    # --- TOMBOL PRINT PDF (SUDAH ADA NAMA) ---
                    st.markdown("---")
                    patient_data_pdf = {
                        "Usia / Gender": f"{int(age)} th / {'Laki-laki' if sex==1 else 'Perempuan'}",
                        "Tekanan Darah": f"{int(trestbps)} mmHg",
                        "Kolesterol": f"{int(chol)} mg/dl",
                        "Gula Darah": "Tinggi (>120)" if fbs==1 else "Normal",
                        "Nyeri Dada": ["Typical", "Atypical", "Non-anginal", "Asymptomatic"][cp-1],
                        "Detak Jantung Max": f"{int(thalach)} bpm",
                        "ST Depression": str(oldpeak)
                    }
                    
                    # Pass 'name' ke fungsi create_pdf
                    pdf_bytes = create_pdf(name, patient_data_pdf, final_result_text, pred_proba, alerts)
                    st.download_button(
                        label=f"📄 CETAK LAPORAN PDF ({name})",
                        data=pdf_bytes,
                        file_name=f"Laporan_{name.replace(' ','_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

                with col_shap: # Atau 'with tab3:' jika di halaman Evaluasi
                    st.subheader("Interpretasi Model (SHAP)")
                    st.caption("Faktor yang paling mempengaruhi prediksi AI:")
                    
                    # 1. BUAT KAMUS TERJEMAHAN (MAPPING)
                    label_mapping = {
                        # Fitur Numerik
                        'age': 'Usia Pasien',
                        'chol': 'Kolesterol (mg/dl)',
                        'trestbps': 'Tekanan Darah (mmHg)',
                        'thalach': 'Detak Jantung Max',
                        'oldpeak': 'Depresi ST (Oldpeak)',
                        'ca': 'Jml Pembuluh Darah Utama',
                        
                        # Fitur Kategorikal (One-Hot Encoded)
                        'sex_1.0': 'Gender: Laki-laki',
                        'sex_0.0': 'Gender: Perempuan',
                        
                        'cp_1.0': 'Nyeri Dada: Typical Angina (Berat)',
                        'cp_2.0': 'Nyeri Dada: Atypical Angina',
                        'cp_3.0': 'Nyeri Dada: Non-Anginal',
                        'cp_4.0': 'Nyeri Dada: Asymptomatic (Tanpa Nyeri)',
                        
                        'exang_1.0': 'Nyeri Olahraga: Ya',
                        'exang_0.0': 'Nyeri Olahraga: Tidak',
                        
                        'fbs_1.0': 'Gula Darah: Tinggi (>120)',
                        'fbs_0.0': 'Gula Darah: Normal',
                        
                        'slope_1.0': 'Slope ST: Naik (Upsloping)',
                        'slope_2.0': 'Slope ST: Datar (Flat)',
                        'slope_3.0': 'Slope ST: Turun (Downsloping)',
                        
                        'thal_3.0': 'Thalassemia: Normal',
                        'thal_6.0': 'Thalassemia: Cacat Tetap',
                        'thal_7.0': 'Thalassemia: Cacat Reversibel',
                        
                        'restecg_0.0': 'EKG: Normal',
                        'restecg_1.0': 'EKG: ST-T Abnormality',
                        'restecg_2.0': 'EKG: LV Hypertrophy'
                    }

                    # 2. TERAPKAN TERJEMAHAN KE NAMA FITUR
                    readable_feat_names = [label_mapping.get(name, name) for name in feat_names]

                    # 3. GENERATE PLOT DENGAN NAMA BARU
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(X_proc)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    shap.summary_plot(
                        shap_vals, 
                        X_proc, 
                        feature_names=readable_feat_names,
                        plot_type="bar", 
                        show=False
                    )

                    plt.xlabel("Tingkat Pengaruh (SHAP Value)")
                    st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

# ==========================================
# 7. HALAMAN VISUALISASI (UI ANDA TETAP)
# ==========================================
elif menu == "Model Performance":
    st.subheader("📊 Analisis Performa Model")
    
    tab1, tab2, tab3 = st.tabs(["Metrik Komparasi", "Confusion Matrix", "Feature Importance"])
    
    with tab1:
        if metrics is not None:
            st.dataframe(metrics.style.format("{:.2%}", subset=['Akurasi', 'Recall', 'Precision', 'F1-Score']))
            
            # Chart Altair
            df_melt = metrics.melt("Model", var_name="Metric", value_name="Score")
            c = alt.Chart(df_melt).mark_bar().encode(
                x='Metric', y='Score', color='Model', column='Model'
            ).properties(height=200)
            st.altair_chart(c)
    
    with tab2:
        if cms is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Decision Tree**")
                if 'Decision Tree' in cms:
                    fig1, ax1 = plt.subplots()
                    sns.heatmap(cms['Decision Tree'], annot=True, fmt='d', cmap='Greys', ax=ax1)
                    st.pyplot(fig1)
            with c2:
                st.write("**XGBoost (Tuned)**")
                if 'XGBoost (Tuned)' in cms:
                    fig2, ax2 = plt.subplots()
                    sns.heatmap(cms['XGBoost (Tuned)'], annot=True, fmt='d', cmap='Blues', ax=ax2)
                    st.pyplot(fig2)
                
    with tab3:
        st.subheader("Interpretasi Model Global (SHAP)")
        st.write("Grafik ini menunjukkan dampak setiap fitur terhadap prediksi secara keseluruhan (Titik Merah = Nilai Tinggi, Biru = Rendah).")

        if st.button("Tampilkan Plot SHAP Global"):
            with st.spinner("Sedang membuat plot (Metode Beeswarm)..."):
                
                # 1. CEK MODEL
                if hasattr(model, 'named_estimators_'):
                    model_for_shap = model.named_estimators_['xgb']
                else:
                    model_for_shap = model

                # 2. MAPPING BAHASA INDONESIA
                label_mapping = {
                    'age': 'Usia Pasien',
                    'chol': 'Kolesterol (mg/dl)',
                    'trestbps': 'Tekanan Darah (mmHg)',
                    'thalach': 'Detak Jantung Max',
                    'oldpeak': 'Depresi ST (Oldpeak)',
                    'ca': 'Jml Pembuluh Darah Utama',
                    'sex_1.0': 'Gender: Laki-laki', 'sex_0.0': 'Gender: Perempuan',
                    'cp_1.0': 'Nyeri: Typical', 'cp_2.0': 'Nyeri: Atypical',
                    'cp_3.0': 'Nyeri: Non-Anginal', 'cp_4.0': 'Nyeri: Asymptomatic',
                    'exang_1.0': 'Nyeri Olahraga: Ya', 'exang_0.0': 'Nyeri Olahraga: Tidak',
                    'fbs_1.0': 'Gula Darah: Tinggi', 'fbs_0.0': 'Gula Darah: Normal',
                    'slope_1.0': 'Slope: Naik', 'slope_2.0': 'Slope: Datar', 'slope_3.0': 'Slope: Turun',
                    'thal_3.0': 'Thal: Normal', 'thal_6.0': 'Thal: Cacat Tetap', 'thal_7.0': 'Thal: Reversibel',
                    'restecg_0.0': 'EKG: Normal', 'restecg_1.0': 'EKG: ST-T Abnorm', 'restecg_2.0': 'EKG: LV Hyper'
                }
                
                # Terapkan terjemahan ke nama fitur
                readable_feat_names = [label_mapping.get(name, name) for name in feat_names]

                # 3. GENERATE PLOT
                try:
                    explainer = shap.TreeExplainer(model_for_shap)
                    shap_values = explainer.shap_values(X_sample)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(
                        shap_values, 
                        X_sample, 
                        feature_names=readable_feat_names, 
                        show=False
                    )
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Gagal memuat SHAP Global: {e}")