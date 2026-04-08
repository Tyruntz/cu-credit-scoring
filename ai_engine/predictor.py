import pandas as pd
import joblib
import numpy as np

# Load artifacts secara global supaya fast response pas API di-hit
try:
    artifacts = joblib.load('ai_engine/credit_model.pkl')
    model = artifacts['model']
    explainer = artifacts['explainer']
    expected_features = artifacts['features']
except Exception as e:
    print(f"[ERROR] credit_model.pkl not found. Please run train_model.py first. Details: {e}")
    model, explainer, expected_features = None, None, None

# Mapping nama variabel jadi bahasa natural
human_labels_map = {
    'usia_tahun': 'Usia Demografi (Tahun)',
    'gaji_per_bulan': 'Kapasitas Pendapatan Bulanan',
    'jumlah_tanggungan': 'Beban Kewajiban Tanggungan Keluarga',
    'riwayat_telat_bayar_hari': 'Rekam Jejak Keterlambatan Pembayaran (SLIK)',
    'lama_bekerja_bulan': 'Stabilitas Durasi Masa Lama Bekerja',
    'rasio_utang_saat_ini': 'Rasio Batas Utang terhadap Pendapatan (DTI Ratio)',
    'nilai_agunan': 'Estimasi Jaminan Likuidasi Nilai Agunan',
    'persentase_kehadiran_rat': 'Tingkat Kehadiran Rapat Anggota Tahunan (RAT)',
    'lulus_diksar': 'Sertifikasi Pendidikan Dasar Koperasi',
    'ada_penjamin': 'Ketersediaan Penjamin/Avalis Kredit'
}

def explain_decision_to_text(shap_explanation_obj, df_input, prediction_prob):
    # Ekstraksi untuk 1 baris observasi (index 0)
    instance_shap = shap_explanation_obj[0]
    shaps = instance_shap.values
    feature_vals = df_input.iloc[0].values
    feature_names = df_input.columns.tolist()
    
    predicted_class = "APPROVED" if prediction_prob >= 0.5 else "REJECTED"
    
    shap_df = pd.DataFrame({
        'Fitur_Independen': feature_names,
        'Nilai_Realitas_Aktual': feature_vals,
        'Metrik_Kontribusi_SHAP_Value': shaps
    })
    
    drivers_positive = shap_df[shap_df['Metrik_Kontribusi_SHAP_Value'] > 0].sort_values(by='Metrik_Kontribusi_SHAP_Value', ascending=False)
    drivers_negative = shap_df[shap_df['Metrik_Kontribusi_SHAP_Value'] < 0].sort_values(by='Metrik_Kontribusi_SHAP_Value', ascending=True)

    # Konstruksi Narasi
    teks = f"STATUS REKOMENDASI SISTEM: {predicted_class}\n"
    teks += f"PROBABILITAS KELAYAKAN: {prediction_prob * 100:.2f}%\n\n"
    
    teks += "RINGKASAN EXECUTIVE (Berdasarkan Analisis Risk Engine):\n"
    
    if predicted_class == "REJECTED":
        teks += "[-] REKOMENDASI: TOLAK. Ditemukan anomali profil risiko tinggi pada parameter berikut:\n"
        for _, row in drivers_negative.head(3).iterrows():
            fitur = human_labels_map.get(row['Fitur_Independen'], row['Fitur_Independen'])
            nilai = format_value(row['Fitur_Independen'], row['Nilai_Realitas_Aktual'])
            teks += f"   > {fitur}: Kondisi saat ini tercatat pada angka {nilai}. Sistem menilai ini sebagai beban berat yang menekan probabilitas kelancaran.\n"
            
        if not drivers_positive.empty:
            teks += "\n[+] Meski begitu, sistem mencatat beberapa poin positif dari profil nasabah ini:\n"
            for _, row in drivers_positive.head(2).iterrows():
                fitur = human_labels_map.get(row['Fitur_Independen'], row['Fitur_Independen'])
                teks += f"   > {fitur} dinilai cukup memadai untuk memberikan stabilitas minor.\n"
                
    else:
        teks += "[+] REKOMENDASI: TERIMA. Profil nasabah memenuhi standar kelayakan kredit dengan dukungan parameter kunci berikut:\n"
        for _, row in drivers_positive.head(3).iterrows():
            fitur = human_labels_map.get(row['Fitur_Independen'], row['Fitur_Independen'])
            nilai = format_value(row['Fitur_Independen'], row['Nilai_Realitas_Aktual'])
            teks += f"   > {fitur}: Angka {nilai} sangat mendukung dan menggaransi probabilitas kapasitas bayar yang tangguh.\n"
            
        if not drivers_negative.empty:
            teks += "\n[-] Catatan Pengawasan Komite (Minor Fluctuation Risk):\n"
            for _, row in drivers_negative.head(2).iterrows():
                fitur = human_labels_map.get(row['Fitur_Independen'], row['Fitur_Independen'])
                teks += f"   > Harap sedikit waspada pada aspek {fitur} sebagai area kelemahan potensial.\n"
                
    return teks

def format_value(feature_name, value):
    """Helper formatting currency & percentage"""
    if 'gaji' in feature_name or 'agunan' in feature_name:
        return f"Rp {value:,.0f}"
    if 'rasio' in feature_name:
        return f"{value*100:.1f}%"
    if 'hari' in feature_name:
        return f"{int(value)} hari"
    if 'persentase' in feature_name:
        return f"{int(value)}%"
    if 'lulus' in feature_name or 'penjamin' in feature_name:
        return "Ya" if value == 1 else "Tidak"
    return str(value)

def analyze_credit_application(input_data_json):
    if model is None or explainer is None:
        return {"error": "Model ML belum di-*load*. Cek file credit_model.pkl."}
        
    # Pastikan susunan kolom presisi sama dengan saat training
    df_input = pd.DataFrame([input_data_json])
    
    # Handling kalau ada kolom kurang
    for col in expected_features:
        if col not in df_input.columns:
            df_input[col] = 0
            
    df_input = df_input[expected_features]
    
    # Prediksi
    prob_array = model.predict_proba(df_input)
    prediction_prob = float(prob_array[0][1]) # Class 1 (Approved) prob
    
    status = "APPROVED" if prediction_prob >= 0.5 else "REJECTED"
    
    # Ekstraksi SHAP
    shap_values_obj = explainer(df_input)
    
    # Generate Natural Language
    text_report = explain_decision_to_text(shap_values_obj, df_input, prediction_prob)
    
    return {
        "status": status,
        "probabilitas": round(prediction_prob, 4),
        "laporan_audit_teks": text_report
    }