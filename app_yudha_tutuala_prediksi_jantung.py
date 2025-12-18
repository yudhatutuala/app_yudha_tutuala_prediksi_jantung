
import time
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

# ------------------------------
# Konfigurasi Halaman
# ------------------------------
st.set_page_config(
    page_title="Prediksi Penyakit Jantung — Versi Peningkatan",
    page_icon="❤️",
    layout="wide"
)

# ------------------------------
# Utilities
# ------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    """Memuat dataset Heart Disease dari sumber DQLab/Google Storage."""
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
    """Melatih model berdasarkan pilihan pengguna dengan pipeline yang tepat.
    - Tree-based: tanpa scaler; PCA opsional (default off).
    - Logistic Regression: scaler + (opsional) PCA.
    - Kalibrasi probabilitas opsional untuk meningkatkan kualitas proba.
    """
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
        # RF umumnya tidak membutuhkan scaling; PCA opsional bila ingin reduksi
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

    # Kalibrasi probabilitas (opsional; berguna terutama untuk tree/RF)
    if calibrate:
        # Pastikan output berupa estimator terkalibrasi yang kompatibel dengan predict_proba
        if isinstance(model, Pipeline):
            calibrated = CalibratedClassifierCV(base_estimator=model, method='sigmoid', cv=5)
            calibrated.fit(X_train, y_train)
            model = calibrated
        else:
            calibrated = CalibratedClassifierCV(base_estimator=model, method='sigmoid', cv=5)
            calibrated.fit(X_train, y_train)
            model = calibrated

    return model

@st.cache_data(show_spinner=False)
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # Cek ketersediaan probabilitas
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
    """Cross-validation cepat (CV=5) pada seluruh dataset untuk gambaran umum stabilitas.
    Mengembalikan rata-rata dan std akurasi.
    """
    X = df.drop('target', axis=1)
    y = df['target']

    # Membangun model sederhana untuk CV (tanpa tuning berat agar cepat)
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

# ------------------------------
# Load Data & Split
# ------------------------------
try:
    heart_data = load_data()
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

X_train, X_test, y_train, y_test = split_data(heart_data)

# ------------------------------
# Sidebar: Pengaturan & Navigasi
# ------------------------------
st.sidebar.title("🏥 Menu Navigasi")
menu = st.sidebar.selectbox(
    "Pilih Menu:",
    ["🏠 Beranda", "ℹ️ Tentang", "📚 Pengenalan Aplikasi", "⚙️ Pengaturan Model", "⚠️ Faktor Risiko", "🔬 Prediksi Penyakit"]
)

# Pengaturan model global (untuk melatih sekali sesuai pilihan)
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Pengaturan Cepat Model")
model_name = st.sidebar.selectbox("Algoritma", ["Decision Tree", "Random Forest", "Logistic Regression"], index=1)
use_pca = st.sidebar.checkbox("Gunakan PCA (reduksi dimensi)", value=False)
pca_components = st.sidebar.slider("Jumlah Komponen PCA", 2, 12, 9, disabled=(not use_pca))
calibrate = st.sidebar.checkbox("Kalibrasi Probabilitas (Platt sigmoid)", value=False)

# Latih model sesuai pengaturan
start = time.time()
model = train_model(model_name, use_pca, pca_components, calibrate, X_train, y_train)
train_time = time.time() - start
metrics = evaluate_model(model, X_test, y_test)
cv_mean, cv_std = cross_val_metrics(model_name, use_pca, pca_components, heart_data)

# ------------------------------
# Halaman: Tentang
# ------------------------------
if menu == "ℹ️ Tentang":
    st.title("ℹ️ Tentang Aplikasi (Versi Peningkatan)")
    st.markdown(f"""
    **Teknologi**: Streamlit, scikit-learn, Plotly, Pandas, NumPy

    **Dataset**: Heart Disease (±303 sampel, 13 fitur + target) — setelah `dropna`

    **Model**: {model_name} (dengan opsi PCA dan kalibrasi probabilitas)

    **Akurasi (Test)**: {metrics['accuracy']:.2%}
    **Precision**: {metrics['precision']:.2%} | **Recall**: {metrics['recall']:.2%} | **F1**: {metrics['f1']:.2%}
    **ROC-AUC**: {metrics['roc_auc']:.3f if metrics['roc_auc'] is not None else '-'}

    **Waktu Latih**: {train_time:.2f} detik
    **CV(5) Akurasi**: {cv_mean:.2%} ± {cv_std:.2%}

    ⚠️ **Disclaimer**: Aplikasi untuk edukasi & referensi, bukan pengganti diagnosis medis profesional.
    """)
    st.markdown("---")
    st.subheader("📄 Classification Report")
    st.text(metrics['report'])

    st.subheader("📊 Confusion Matrix")
    cm = metrics['cm']
    fig_cm = px.imshow(cm,
                       labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                       x=['Tidak Sakit', 'Sakit'],
                       y=['Tidak Sakit', 'Sakit'],
                       text_auto=True,
                       color_continuous_scale='Blues')
    fig_cm.update_layout(title="Confusion Matrix (Test)")
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("🥧 Distribusi Target")
    target_counts = heart_data['target'].value_counts()
    fig_dist = go.Figure(data=[go.Pie(
        labels=['Tidak Sakit', 'Sakit'],
        values=target_counts.values,
        hole=0.4,
        marker=dict(colors=['#2ecc71', '#e74c3c'])
