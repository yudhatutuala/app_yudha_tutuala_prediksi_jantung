# -*- coding: utf-8 -*-
import time
import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import plotly.express as px
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="Prediksi Penyakit Jantung — Versi Peningkatan (Save/Load)",
    page_icon="❤️",
    layout="wide"
)

@st.cache_data(show_spinner=False)
def load_data():
    local_path = 'heart_disease.csv'
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
    else:
        url = 'https://storage.googleapis.com/dqlab-dataset/heart_disease.csv'
        df = pd.read_csv(url)
    df = df.dropna()
    return df

@st.cache_data(show_spinner=False)
def split_data(df, test_size=0.2, random_state=42):
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

@st.cache_data(show_spinner=False)
def train_model(model_name: str, use_pca: bool, pca_components: int, calibrate: bool, X_train: pd.DataFrame, y_train: pd.Series):
    if model_name == 'Decision Tree':
        base = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'clf__max_depth': [None, 4, 6, 8, 10, 12],
            'clf__min_samples_split': [2, 5, 10],
            'clf__min_samples_leaf': [1, 2, 4],
            'clf__criterion': ['gini', 'entropy']
        }
        steps = []
        if use_pca:
            steps.extend([('scaler', StandardScaler()), ('pca', PCA(n_components=pca_components))])
        steps.append(('clf', base))
        pipe = Pipeline(steps)
        grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    elif model_name == 'Random Forest':
        base = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1, random_state=42, n_jobs=-1)
        steps = []
        if use_pca:
            steps.extend([('scaler', StandardScaler()), ('pca', PCA(n_components=pca_components))])
        steps.append(('clf', base))
        model = Pipeline(steps)
        model.fit(X_train, y_train)
    elif model_name == 'Logistic Regression':
        base = LogisticRegression(max_iter=2000, solver='lbfgs')
        steps = [('scaler', StandardScaler())]
        if use_pca:
            steps.append(('pca', PCA(n_components=pca_components)))
        steps.append(('clf', base))
        model = Pipeline(steps)
        model.fit(X_train, y_train)
    else:
        raise ValueError('Model tidak dikenal.')

    if calibrate:
        # gunakan estimator= sesuai versi scikit-learn terbaru
        calibrated = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=5)
        calibrated.fit(X_train, y_train)
        model = calibrated

    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        y_proba = None
        roc_auc = None
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'cm': cm,
        'report': report,
        'y_pred': y_pred,
        'y_proba': y_proba
    }

@st.cache_data(show_spinner=False)
def cross_val_metrics(model_name: str, use_pca: bool, pca_components: int, df: pd.DataFrame):
    X = df.drop('target', axis=1)
    y = df['target']
    if model_name == 'Decision Tree':
        steps = []
        if use_pca:
            steps.extend([('scaler', StandardScaler()), ('pca', PCA(n_components=pca_components))])
        steps.append(('clf', DecisionTreeClassifier(max_depth=8, random_state=42)))
        est = Pipeline(steps)
    elif model_name == 'Random Forest':
        steps = []
        if use_pca:
            steps.extend([('scaler', StandardScaler()), ('pca', PCA(n_components=pca_components))])
        steps.append(('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)))
        est = Pipeline(steps)
    else:
        steps = [('scaler', StandardScaler())]
        if use_pca:
            steps.append(('pca', PCA(n_components=pca_components)))
        steps.append(('clf', LogisticRegression(max_iter=2000)))
        est = Pipeline(steps)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        est.fit(X_tr, y_tr)
        accs.append(accuracy_score(y_te, est.predict(X_te)))
    return np.mean(accs), np.std(accs)

# Load data
try:
    heart_data = load_data()
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

X_train, X_test, y_train, y_test = split_data(heart_data)

# Sidebar & Pengaturan
st.sidebar.title("🏥 Menu Navigasi")
menu = st.sidebar.selectbox(
    "Pilih Menu:",
    ["🏠 Beranda", "ℹ️ Tentang", "📚 Pengenalan Aplikasi", "⚙️ Pengaturan Model", "⚠️ Faktor Risiko", "🔬 Prediksi Penyakit"]
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Pengaturan Cepat Model")
model_name = st.sidebar.selectbox("Algoritma", ["Decision Tree", "Random Forest", "Logistic Regression"], index=1)
use_pca = st.sidebar.checkbox("Gunakan PCA (reduksi dimensi)", value=False)
pca_components = st.sidebar.slider("Jumlah Komponen PCA", 2, 12, 9, disabled=(not use_pca))
calibrate = st.sidebar.checkbox("Kalibrasi Probabilitas (Platt sigmoid)", value=False)

# Tombol Save/Muat model
save_btn = st.sidebar.button("💾 Simpan Model")
load_btn = st.sidebar.button("📂 Muat Model")

# Latih model sesuai pengaturan
start = time.time()
model = train_model(model_name, use_pca, pca_components, calibrate, X_train, y_train)
train_time = time.time() - start
metrics = evaluate_model(model, X_test, y_test)
cv_mean, cv_std = cross_val_metrics(model_name, use_pca, pca_components, heart_data)

# Simpan/Muat
if save_btn:
    try:
        joblib.dump(model, "model_terbaik.pkl")
        st.sidebar.success("Model disimpan: model_terbaik.pkl")
    except Exception as e:
        st.sidebar.error(f"Gagal menyimpan model: {e}")

if load_btn:
    try:
        model = joblib.load("model_terbaik.pkl")
        st.sidebar.success("Model dimuat dari: model_terbaik.pkl")
        metrics = evaluate_model(model, X_test, y_test)
    except Exception as e:
        st.sidebar.error(f"Gagal memuat model: {e}")

# Halaman: Tentang
if menu == "ℹ️ Tentang":
    st.title("ℹ️ Tentang Aplikasi (Versi Save/Load)")
    roc_text = "-"
    if metrics['roc_auc'] is not None:
        roc_text = f"{metrics['roc_auc']:.3f}"
    st.markdown(
        f"""
**Teknologi**: Streamlit, scikit-learn, Plotly, Pandas, NumPy

**Dataset**: Heart Disease (±303 sampel, 13 fitur + target) — setelah `dropna`

**Model**: {model_name} (opsi PCA & kalibrasi probabilitas)

**Akurasi (Test)**: {metrics['accuracy']:.2%}  
**Precision**: {metrics['precision']:.2%} | **Recall**: {metrics['recall']:.2%} | **F1**: {metrics['f1']:.2%}  
**ROC-AUC**: {roc_text}

**Waktu Latih**: {train_time:.2f} detik  
**CV(5) Akurasi**: {cv_mean:.2%} ± {cv_std:.2%}

⚠️ **Disclaimer**: Aplikasi untuk edukasi & referensi, bukan pengganti diagnosis medis profesional.
        """
    )
    st.markdown("---")
    st.subheader("📄 Classification Report")
    st.text(metrics['report'])
    st.subheader("📊 Confusion Matrix")
    cm = metrics['cm']
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
        x=['Tidak Sakit', 'Sakit'],
        y=['Tidak Sakit', 'Sakit'],
        text_auto=True,
        color_continuous_scale='Blues'
    )
    fig_cm.update_layout(title="Confusion Matrix (Test)")
    st.plotly_chart(fig_cm, use_container_width=True)
    st.subheader("🥧 Distribusi Target")
    target_counts = heart_data['target'].value_counts()
    fig_dist = go.Figure(
        data=[
            go.Pie(
                labels=['Tidak Sakit', 'Sakit'],
                values=target_counts.values,
                hole=0.4,
                marker=dict(colors=['#2ecc71', '#e74c3c'])
            )
        ]
    )
    fig_dist.update_layout(title="Distribusi Target — Seluruh Dataset")
    st.plotly_chart(fig_dist, use_container_width=True)

# Halaman: Pengenalan
elif menu == "📚 Pengenalan Aplikasi":
    st.title("📚 Pengenalan Aplikasi")
    st.markdown(
        """
**Cara Kerja (Versi Save/Load)**
1) Muat data → bersihkan `NaN` → pisah fitur & target (stratified split)
2) Bangun **Pipeline** sesuai model (scaler/PCA bila diperlukan)
3) Latih model + (opsional) **kalibrasi probabilitas**
4) Evaluasi: akurasi, precision, recall, F1, ROC-AUC, confusion matrix
5) Prediksi interaktif + ringkasan input & grafik probabilitas
6) **Simpan/muat model** untuk penggunaan berulang
        """
    )
    st.info("💡 Gunakan tombol **Simpan/Muat model** di sidebar.")

# Halaman: Pengaturan Model
elif menu == "⚙️ Pengaturan Model":
    st.title("⚙️ Pengaturan Model")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Akurasi (Test)", f"{metrics['accuracy']:.2%}")
    col2.metric("Precision", f"{metrics['precision']:.2%}")
    col3.metric("Recall", f"{metrics['recall']:.2%}")
    col4.metric("F1", f"{metrics['f1']:.2%}")
    col5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}" if metrics['roc_auc'] is not None else "-")
    st.markdown("---")
    st.subheader("📊 Confusion Matrix")
    cm = metrics['cm']
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
        x=['Tidak Sakit', 'Sakit'],
        y=['Tidak Sakit', 'Sakit'],
        text_auto=True,
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    if metrics['y_proba'] is not None:
        st.markdown("---")
        st.subheader("📈 Distribusi Probabilitas (Test)")
        fig_hist = px.histogram(metrics['y_proba'], nbins=20, title="Histogram Probabilitas Kelas 'Sakit'",
                                labels={'value': 'Probabilitas'}, opacity=0.8)
        st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown("---")
    st.write(f"⏱️ Waktu Latih: **{train_time:.2f} detik** | CV(5) Akurasi: **{cv_mean:.2%} ± {cv_std:.2%}**")

# Halaman: Faktor Risiko
elif menu == "⚠️ Faktor Risiko":
    st.title("⚠️ Faktor Risiko Penyakit Jantung")
    st.markdown(
        """
**Tidak Dapat Diubah**: Usia, jenis kelamin, riwayat keluarga, faktor genetik/etnis.

**Dapat Diubah**: Tekanan darah tinggi, kolesterol tinggi, merokok, obesitas, diabetes,
kurang aktivitas fisik, stres kronis, konsumsi alkohol berlebihan.

👉 Aplikasi ini bersifat edukasi; konsultasikan ke tenaga medis profesional untuk diagnosis.
        """
    )

# Halaman: Prediksi Interaktif
elif menu == "🔬 Prediksi Penyakit":
    st.title("🔬 Sistem Prediksi Penyakit Jantung")
    st.markdown("Masukkan parameter medis pasien di sidebar untuk mendapatkan prediksi risiko.")

    st.sidebar.header("📋 Data Pasien")
    age = st.sidebar.slider("Usia", 20, 100, 50)
    sex = st.sidebar.selectbox("Jenis Kelamin", options=[0, 1], format_func=lambda x: "Wanita" if x == 0 else "Pria")
    cp = st.sidebar.selectbox("Tipe Nyeri Dada", options=[0, 1, 2, 3],
                               format_func=lambda x: ["Angina Tipikal", "Angina Atipikal", "Nyeri Non-anginal", "Asimtomatik"][x])
    trestbps = st.sidebar.slider("Tekanan Darah Istirahat (mmHg)", 80, 200, 120)
    chol = st.sidebar.slider("Kolesterol Serum (mg/dL)", 100, 400, 200)
    fbs = st.sidebar.selectbox("Gula Darah Puasa > 120 mg/dL", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
    restecg = st.sidebar.selectbox("Hasil EKG Istirahat", options=[0, 1, 2],
                                   format_func=lambda x: ["Normal", "Kelainan Gelombang ST-T", "Hipertrofi Ventrikel Kiri"][x])
    thalach = st.sidebar.slider("Detak Jantung Maksimum", 60, 220, 150)
    exang = st.sidebar.selectbox("Angina saat Olahraga", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.5, 1.0, 0.1)
    slope = st.sidebar.selectbox("Slope Segmen ST", options=[0, 1, 2], format_func=lambda x: ["Naik", "Datar", "Turun"][x])
    ca = st.sidebar.selectbox("Jumlah Pembuluh Darah Mayor (0-3)", options=[0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia", options=[0, 1, 2, 3],
                                format_func=lambda x: ["Normal", "Cacat Tetap", "Cacat Reversibel", "Tidak Diketahui"][x])

    predict_button = st.sidebar.button("🔍 Prediksi", use_container_width=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📊 Performa Model (Test)")
        st.metric("Akurasi", f"{metrics['accuracy']:.2%}")
        st.metric("Precision", f"{metrics['precision']:.2%}")
        st.metric("Recall", f"{metrics['recall']:.2%}")
        st.metric("F1", f"{metrics['f1']:.2%}")
        if metrics['roc_auc'] is not None:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        cm = metrics['cm']
        fig_cm = px.imshow(cm,
                           labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                           x=['Tidak Sakit', 'Sakit'],
                           y=['Tidak Sakit', 'Sakit'],
                           text_auto=True,
                           color_continuous_scale='Blues')
        fig_cm.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.subheader("📈 Ringkasan Dataset")
        target_counts = heart_data['target'].value_counts()
        fig_dist = go.Figure(data=[go.Pie(labels=['Tidak Sakit', 'Sakit'], values=target_counts.values, hole=0.4,
                                          marker=dict(colors=['#2ecc71', '#e74c3c']))])
        fig_dist.update_layout(title="Distribusi Target")
        st.plotly_chart(fig_dist, use_container_width=True)
        st.metric("Total Sampel", len(heart_data))
        st.metric("Jumlah Fitur", len(heart_data.columns) - 1)

    if predict_button:
        st.markdown("---")
        st.subheader("🎯 Hasil Prediksi")
        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                    thalach, exang, oldpeak, slope, ca, thal]],
                                  columns=X_train.columns)
        pred_class = model.predict(input_data)[0]
        try:
            pred_proba = model.predict_proba(input_data)[0]
        except Exception:
            pred_proba = None
        colr1, colr2, colr3 = st.columns([1, 2, 1])
        with colr2:
            if pred_class == 1:
                st.error("⚠️ RISIKO TINGGI: Terdeteksi Penyakit Jantung")
                if pred_proba is not None:
                    st.markdown(f"**Tingkat Kepercayaan:** {pred_proba[1]:.1%}")
                st.markdown("""
### 📋 Rekomendasi:
- 🏥 **Segera konsultasi ke dokter spesialis jantung**
- 💊 Ikuti pengobatan yang diresepkan
- 🥗 Terapkan pola hidup sehat
- 📊 Monitoring rutin diperlukan
- 🚨 Hindari aktivitas berat tanpa pengawasan medis
                """)
            else:
                st.success("✅ RISIKO RENDAH: Tidak Terdeteksi Penyakit Jantung")
                if pred_proba is not None:
                    st.markdown(f"**Tingkat Kepercayaan:** {pred_proba[0]:.1%}")
                st.markdown("""
### 📋 Rekomendasi:
- 💚 Lanjutkan gaya hidup sehat
- 🏃 Olahraga teratur (150 menit/minggu)
- 🥗 Konsumsi makanan bergizi seimbang
- 🩺 Cek kesehatan rutin setiap 6–12 bulan
- 🚭 Hindari merokok dan alkohol berlebihan
                """)

        if pred_proba is not None:
            st.markdown("---")
            st.subheader("📊 Probabilitas Prediksi")
            fig_proba = go.Figure(data=[go.Bar(x=['Tidak Sakit', 'Sakit'], y=pred_proba,
                                               marker=dict(color=['#2ecc71', '#e74c3c']),
                                               text=[f'{p:.1%}' for p in pred_proba], textposition='auto')])
            fig_proba.update_layout(yaxis_title="Probabilitas", yaxis=dict(range=[0, 1]), showlegend=False)
            st.plotly_chart(fig_proba, use_container_width=True)

        st.markdown("---")
        st.subheader("📝 Ringkasan Data Input")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
- **Usia:** {age} tahun
- **Jenis Kelamin:** {'Pria' if sex == 1 else 'Wanita'}
- **Tekanan Darah:** {trestbps} mmHg
- **Kolesterol:** {chol} mg/dL
- **Gula Darah Puasa:** {'Ya' if fbs == 1 else 'Tidak'}
            """)
        with c2:
            st.markdown(f"""
- **Detak Jantung Max:** {thalach} bpm
- **Angina saat Olahraga:** {'Ya' if exang == 1 else 'Tidak'}
- **ST Depression:** {oldpeak}
- **Jumlah Pembuluh:** {ca}
            """)
        with c3:
            tipe_cp = ["Angina Tipikal", "Angina Atipikal", "Nyeri Non-anginal", "Asimtomatik"][cp]
            tipe_restecg = ["Normal", "Kelainan ST-T", "Hipertrofi LV"][restecg]
            tipe_slope = ["Naik", "Datar", "Turun"][slope]
            tipe_thal = ["Normal", "Cacat Tetap", "Cacat Reversibel", "Tidak Diketahui"][thal]
            st.markdown(f"""
- **Tipe Nyeri Dada:** {tipe_cp}
- **Hasil EKG:** {tipe_restecg}
- **Slope ST:** {tipe_slope}
- **Thalassemia:** {tipe_thal}
            """)

# Beranda
else:
    st.title("❤️ Sistem Prediksi Penyakit Jantung — Versi Save/Load")
    st.markdown("""
Aplikasi ini menggunakan **Machine Learning** untuk memprediksi risiko penyakit jantung.
Versi ini menambahkan tombol **Simpan/Muat model** agar praktis digunakan berulang.
    """)
    colA, colB, colC = st.columns(3)
    with colA:
        st.info(f"""
### 🎯 Lebih Akurat
Model: **{model_name}** | Akurasi (test): **{metrics['accuracy']:.1%}**
        """)
    with colB:
        st.success("""
### 🚀 Lebih Andal
Stratified split, cross-validation, dan pipeline terintegrasi
        """)
    with colC:
        st.warning("""
### 📊 Lebih Informatif
ROC-AUC, confusion matrix, classification report, histogram proba
        """)
    st.markdown("---")
    st.subheader("📊 Statistik Model")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Akurasi (Test)", f"{metrics['accuracy']:.1%}")
    c2.metric("CV(5) Akurasi", f"{cv_mean:.1%} ± {cv_std:.1%}")
    c3.metric("Algoritma", model_name)
    c4.metric("Waktu Latih", f"{train_time:.2f} detik")
    st.markdown("---")
    st.error("""
### ⚠️ DISCLAIMER PENTING
Aplikasi ini **HANYA** untuk edukasi & referensi. Hasil tidak dapat menggantikan diagnosis medis profesional.
Selalu konsultasikan dengan dokter atau tenaga medis yang berkualifikasi untuk keputusan klinis.
    """)
    st.markdown("---")
    st.success("""
### 💡 Siap Memulai?
Gunakan menu di sidebar:
- **ℹ️ Tentang** — Info aplikasi & metrik
- **📚 Pengenalan Aplikasi** — Alur kerja
- **⚙️ Pengaturan Model** — Eksperimen algoritma & PCA
- **⚠️ Faktor Risiko** — Edukasi singkat
- **🔬 Prediksi Penyakit** — Mulai prediksi sekarang!
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
<p>⚕️ Aplikasi edukasi. Konsultasikan dengan tenaga medis profesional untuk diagnosis yang akurat.</p>
<p>© 2025 Sistem Prediksi Penyakit Jantung — Versi Save/Load • Dibuat dengan ❤️ menggunakan Streamlit</p>
</div>
""", unsafe_allow_html=True)
