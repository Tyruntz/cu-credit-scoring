import os
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import joblib
from scipy.special import expit
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# 1. METODOLOGI GENERASI DATASET DUMMY
def generate_koperasi_dataset(n_samples=1000, random_state=42):
    np.random.seed(random_state)
    
    # Fitur 5C Konvensional
    usia = np.clip(np.random.normal(loc=38, scale=10, size=n_samples), 21, 65).astype(int)
    gaji = np.round(np.random.lognormal(mean=15.2, sigma=0.5, size=n_samples) / 100000) * 100000
    tanggungan = np.random.poisson(lam=2, size=n_samples)
    telat_bayar = np.random.choice([0, 30, 60, 90, 150, 300], p=[0.55, 0.20, 0.10, 0.08, 0.05, 0.02], size=n_samples)
    lama_bekerja = np.random.randint(6, 120, size=n_samples)
    rasio_utang = np.random.uniform(0.1, 0.7, size=n_samples)
    nilai_agunan = np.round((gaji * np.random.uniform(3, 15, size=n_samples)) / 500000) * 500000

    # Fitur Tambahan: Kultural Koperasi
    persentase_kehadiran_rat = np.random.randint(0, 101, size=n_samples) # 0-100%
    lulus_diksar = np.random.choice([0, 1], p=[0.4, 0.6], size=n_samples) # 0: Belum, 1: Sudah
    ada_penjamin = np.random.choice([0, 1], p=[0.7, 0.3], size=n_samples) # 0: Tidak, 1: Ada Penjamin
    
    df = pd.DataFrame({
        'usia_tahun': usia,
        'gaji_per_bulan': gaji,
        'jumlah_tanggungan': tanggungan,
        'riwayat_telat_bayar_hari': telat_bayar,
        'lama_bekerja_bulan': lama_bekerja,
        'rasio_utang_saat_ini': rasio_utang,
        'nilai_agunan': nilai_agunan,
        'persentase_kehadiran_rat': persentase_kehadiran_rat,
        'lulus_diksar': lulus_diksar,
        'ada_penjamin': ada_penjamin
    })
    
    # Agregat parameter skor kelayakan indeks (Latent Score)
    norm_gaji = np.log1p(df['gaji_per_bulan'])
    
    skor_laten = (
        (norm_gaji * 1.5) -
        (df['jumlah_tanggungan'] * 0.4) -
        (df['riwayat_telat_bayar_hari'] * 0.05) + 
        (df['lama_bekerja_bulan'] * 0.01) -
        (df['rasio_utang_saat_ini'] * 3.0) + 
        (np.log1p(df['nilai_agunan']) * 0.5) +
        (df['persentase_kehadiran_rat'] * 0.015) +  # Boost tipis kalau rajin RAT
        (df['lulus_diksar'] * 0.8) +                # Bukti udah diedukasi
        (df['ada_penjamin'] * 1.5)                  # Mitigasi risiko lapis 3
    )
    
    # White noise
    noise = np.random.normal(0, 1.5, n_samples)
    skor_laten += noise
    
    # Sigmoid to Prob
    prob_approval = expit(skor_laten - np.median(skor_laten))
    df['target_label'] = (prob_approval >= 0.5).astype(int)
    
    return df

# 2. TRAINING & SAVING
if __name__ == "__main__":
    print("Mulai *generate* dataset kultural koperasi...")
    df = generate_koperasi_dataset(1500)
    
    X = df.drop('target_label', axis=1)
    y = df['target_label']
    
    print("Training XGBoost Classifier...")
    # Train
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X, y)
    
    print("Inisialisasi SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Save objects
    artifacts = {
        'model': model,
        'explainer': explainer,
        'features': X.columns.tolist()
    }
    
    # Buat direktori jika belum ada
    os.makedirs('ai_engine', exist_ok=True)
    
    joblib.dump(artifacts, 'ai_engine/credit_model.pkl')
    print("Training *Done*! Model dan Explainer tersimpan di ai_engine/credit_model.pkl")